/// Renderer-agnostic graph model and builder
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use baml_types::BamlMap;
use internal_baml_diagnostics::SerializedSpan;

use crate::ast::{
    header_collector::{HeaderLabelKind, Hid},
    HeaderIndex, RenderableHeader, ScopeId,
};

/// Config: maximum number of direct children (markdown + nested) a non-branching
/// container may have to be flattened into a linear sequence instead of a subgraph.
///
/// For example, with `1` (default), containers with 0 or 1 child are flattened.
/// Top-level scopes are never flattened if they have any children, regardless of this value.
const MAX_CHILDREN_TO_FLATTEN: usize = 1;

pub fn build<'index>(
    index: &'index HeaderIndex,
    config: BuilderConfig,
) -> (Graph<'index>, BamlMap<NodeId, SerializedSpan>) {
    let pre = Prelude::from_index(index);
    let builder = GraphBuilder::new(index, &pre, config);
    let (graph, span_map) = builder.build();
    (graph, span_map)
}

use super::{graph, SHOW_CALL_NODES};

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum Direction {
    TD,
    LR,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum NodeKind<'index> {
    Header(Hid, Option<SerializedSpan>),
    Decision(Hid, Option<SerializedSpan>),
    Call { header: Hid, callee: &'index str },
}

#[derive(Debug, Clone)]
pub struct Node<'index> {
    pub id: NodeId,
    pub label: &'index str,
    pub kind: NodeKind<'index>,
    pub cluster: Option<ClusterId>,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
}
#[derive(Debug, Clone)]
pub struct Cluster<'index> {
    pub id: ClusterId,
    pub label: &'index str,
    pub parent: Option<ClusterId>,
}

#[derive(Debug, Default)]
pub struct Graph<'index> {
    pub nodes: Vec<Node<'index>>,
    pub edges: Vec<Edge>,
    pub clusters: Vec<Cluster<'index>>,
}

#[derive(Debug, Clone, Copy)]
pub struct BuilderConfig {
    show_call_nodes: bool,
}

impl Default for BuilderConfig {
    fn default() -> Self {
        Self {
            show_call_nodes: SHOW_CALL_NODES,
        }
    }
}

struct GraphBuilder<'index, 'pre> {
    index: &'index HeaderIndex,
    cfg: BuilderConfig,
    graph: Graph<'index>,
    next_node: u32,
    next_cluster: u32,
    by_hid: &'pre HashMap<Hid, &'index RenderableHeader>,
    md_children: &'pre HashMap<Hid, Vec<Hid>>,
    has_md_parent: &'pre HashSet<Hid>,
    nested_children: &'pre HashMap<Hid, Vec<Hid>>,
    // TODO: check visit order
    header_entry: HashMap<Hid, NodeId>,
    header_exits: HashMap<Hid, Vec<NodeId>>,
    // we're going to need stable iteration in snapshot tests.
    span_map: BamlMap<NodeId, SerializedSpan>,
    call_node_cache: HashMap<(Hid, &'index str), NodeId>,
}

/// Cached, precomuted data used by graph builder.
struct Prelude<'index> {
    // NOTE: this could be a Box<[&'index RenderableHeader]>. Doesn't matter much.
    /// Map to reference by `Hid`, since [`HeaderIndex::headers`] has a different order.
    by_hid: HashMap<Hid, &'index RenderableHeader>,
    /// Nodes that have markdown header children, & their respective children.
    md_children: HashMap<Hid, Vec<Hid>>,
    /// Set of nodes that have a markdown header parent.
    has_md_parent: HashSet<Hid>,
    // TODO: check if this can be merged with `md_children`
    /// For nested edges, the ones that cross a code scope.
    nested_children: HashMap<Hid, Vec<Hid>>,
}

impl<'index> Prelude<'index> {
    pub fn from_index(index: &'index HeaderIndex) -> Self {
        let mut by_hid = HashMap::new();

        let mut idstr_to_hid = HashMap::new();
        for h in &index.headers {
            by_hid.insert(h.hid, h);
            idstr_to_hid.insert(h.id.as_str(), h.hid);
        }

        let mut md_children: HashMap<_, Vec<_>> = HashMap::new();
        let mut has_md_parent = HashSet::new();
        for h in &index.headers {
            if let Some(pid) = &h.parent_id {
                if let Some(&ph) = idstr_to_hid.get(pid.as_str()) {
                    let parent = by_hid[&ph];
                    if parent.scope == h.scope {
                        md_children.entry(ph).or_default().push(h.hid);
                        has_md_parent.insert(h.hid);
                    }
                }
            }
        }

        let mut nested_children: HashMap<_, Vec<_>> = HashMap::new();

        for (p, c) in nested_scope_edges(index, &by_hid) {
            nested_children.entry(p).or_default().push(c);
        }

        Self {
            by_hid,
            md_children,
            has_md_parent,
            nested_children,
        }
    }
}

impl<'index, 'pre> GraphBuilder<'index, 'pre> {
    pub fn new(index: &'index HeaderIndex, pre: &'pre Prelude<'index>, cfg: BuilderConfig) -> Self {
        Self {
            index,
            cfg,
            graph: Graph::default(),
            next_node: 0,
            next_cluster: 0,
            by_hid: &pre.by_hid,
            md_children: &pre.md_children,
            has_md_parent: &pre.has_md_parent,
            nested_children: &pre.nested_children,
            header_entry: HashMap::new(),
            header_exits: HashMap::new(),
            span_map: BamlMap::new(),
            call_node_cache: HashMap::new(),
        }
    }

    pub fn build(mut self) -> (Graph<'index>, BamlMap<NodeId, SerializedSpan>) {
        let mut tops: Vec<(String, usize, usize, ScopeId)> = Vec::new();
        let mut seen_scopes: HashSet<ScopeId> = HashSet::new();

        let scope_root = build_scope_roots(self.index);
        // NOTE: the check is negated (!contains), but there's no easy way yet to get
        // all non-nested targets. Building the opposite set requires iterating through the nested
        // targets anyway.
        let nested_targets: HashSet<_> = all_nested_targets(self.index, &self.by_hid).collect();

        for h in &self.index.headers {
            if seen_scopes.insert(h.scope) {
                let root_hid = scope_root[&h.scope];
                if !nested_targets.contains(&root_hid) {
                    let root = self.by_hid[&root_hid];
                    tops.push((
                        root.span.file.path_buf().to_string_lossy().into_owned(),
                        root.span.start,
                        root.span.end,
                        h.scope,
                    ));
                }
            }
        }
        tops.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

        let mut visited_scopes: HashSet<ScopeId> = HashSet::new();
        for (_, _, _, scope) in tops {
            self.build_scope_sequence(scope, &mut visited_scopes, None);
        }
        (self.graph, self.span_map)
    }

    fn build_scope_sequence(
        &mut self,
        scope: ScopeId,
        visited_scopes: &mut HashSet<ScopeId>,
        parent_cluster: Option<ClusterId>,
    ) {
        if !visited_scopes.insert(scope) {
            return;
        }
        let items: Vec<Hid> = self
            .index
            .headers_in_scope_iter(scope)
            .filter(|h| !self.has_md_parent.contains(&h.hid))
            .map(|h| h.hid)
            .collect();

        let mut prev_exits = None;
        for hid in items {
            let (entry, exits) = self.build_header(hid, visited_scopes, parent_cluster);
            if let Some(prev) = prev_exits.take() {
                for e in prev {
                    self.graph.edges.push(Edge { from: e, to: entry });
                }
            }
            prev_exits = Some(exits);
        }
    }

    // NOTE: we may wont to make the caches have more lifetime
    // than build so that we can return &'cache [NodeId] after restoring.
    fn build_header(
        &mut self,
        hid: Hid,
        visited_scopes: &mut HashSet<ScopeId>,
        parent_cluster: Option<ClusterId>,
    ) -> (NodeId, Vec<NodeId>) {
        if let Some(entry) = self.header_entry.get(&hid).copied() {
            let exits = self
                .header_exits
                .get(&hid)
                .cloned()
                .unwrap_or_else(|| vec![entry]);
            return (entry, exits);
        }
        let header = self.by_hid[&hid];
        let md_children = self.md_children.get(&hid).cloned().unwrap_or_default();
        let nested_children = self.nested_children.get(&hid).cloned().unwrap_or_default();
        let has_md = !md_children.is_empty();
        let has_nested = !nested_children.is_empty();
        let is_branching = header.label_kind == HeaderLabelKind::If;

        if !has_md && !has_nested {
            let node_id = self.new_node_id();
            let span = SerializedSpan::serialize(&header.span);
            self.span_map.insert(node_id, span.clone());
            self.graph.nodes.push(Node {
                id: node_id,
                label: header.title.as_ref(),
                kind: NodeKind::Header(hid, Some(span)),
                cluster: parent_cluster,
            });
            self.header_entry.insert(hid, node_id);
            self.header_exits.insert(hid, vec![node_id]);
            if self.cfg.show_call_nodes {
                self.render_calls_for_header(hid, node_id, parent_cluster);
            }
            return (node_id, vec![node_id]);
        }

        let total_children = md_children.len() + nested_children.len();
        let mut single_nested_child_has_multiple_items = false;
        if nested_children.len() == 1 {
            let child_root = self.by_hid[&nested_children[0]];
            let count = self.index.headers_in_scope_iter(child_root.scope).count();
            single_nested_child_has_multiple_items = count > 1;
        }
        let should_flatten = !is_branching
            && total_children <= MAX_CHILDREN_TO_FLATTEN
            && !(nested_children.len() == 1 && single_nested_child_has_multiple_items);

        if should_flatten {
            let node_id = self.new_node_id();
            let span = SerializedSpan::serialize(&header.span);
            self.span_map.insert(node_id, span.clone());
            let kind = if is_branching {
                NodeKind::Decision(hid, Some(span.clone()))
            } else {
                NodeKind::Header(hid, Some(span.clone()))
            };
            self.graph.nodes.push(Node {
                id: node_id,
                label: header.title.as_ref(),
                kind,
                cluster: parent_cluster,
            });
            self.header_entry.insert(hid, node_id);
            let mut exits = vec![node_id];
            if md_children.len() == 1 {
                let (c_entry, c_exits) =
                    self.build_header(md_children[0], visited_scopes, parent_cluster);
                self.graph.edges.push(Edge {
                    from: node_id,
                    to: c_entry,
                });
                exits = c_exits;
            } else if nested_children.len() == 1 {
                let child_root_hid = nested_children[0];
                let child_scope = self.by_hid[&child_root_hid].scope;
                self.build_scope_sequence(child_scope, visited_scopes, parent_cluster);
                let (c_entry, c_exits) =
                    self.build_header(child_root_hid, visited_scopes, parent_cluster);
                self.graph.edges.push(Edge {
                    from: node_id,
                    to: c_entry,
                });
                exits = c_exits;
            }
            self.header_exits.insert(hid, exits.clone());
            return (node_id, exits);
        }

        if is_branching {
            let cluster_id = self.new_cluster_id();
            self.graph.clusters.push(Cluster {
                id: cluster_id,
                label: header.title.as_ref(),
                parent: parent_cluster,
            });

            let decision_id = self.new_node_id();
            let span = SerializedSpan::serialize(&header.span);
            self.span_map.insert(decision_id, span.clone());
            self.graph.nodes.push(Node {
                id: decision_id,
                label: header.title.as_ref(),
                kind: NodeKind::Decision(hid, Some(span)),
                cluster: Some(cluster_id),
            });
            self.header_entry.insert(hid, decision_id);
            if self.cfg.show_call_nodes {
                self.render_calls_for_header(hid, decision_id, Some(cluster_id));
            }

            let mut branch_exits = Vec::new();
            for child_root_hid in nested_children.iter() {
                let child_scope = self.by_hid[child_root_hid].scope;
                self.build_scope_sequence(child_scope, visited_scopes, Some(cluster_id));
                let (entry, exits) =
                    self.build_header(*child_root_hid, visited_scopes, Some(cluster_id));
                self.graph.edges.push(Edge {
                    from: decision_id,
                    to: entry,
                });
                if self.cfg.show_call_nodes {
                    self.render_calls_for_header(*child_root_hid, entry, Some(cluster_id));
                }
                branch_exits.extend(exits);
            }

            // NOTE: here we discard the Vec<>  from `build_header`!
            let md_ids_only: Vec<_> = md_children
                .iter()
                .copied()
                .map(|ch| self.build_header(ch, visited_scopes, Some(cluster_id)).0)
                .collect();

            if let Some(first_md) = md_ids_only.first().copied() {
                if !branch_exits.is_empty() {
                    for e in branch_exits.iter() {
                        self.graph.edges.push(Edge {
                            from: *e,
                            to: first_md,
                        });
                    }
                } else {
                    self.graph.edges.push(Edge {
                        from: decision_id,
                        to: first_md,
                    });
                }
            }
            for win in md_ids_only.windows(2) {
                let (a, b) = (&win[0], &win[1]);
                self.graph.edges.push(Edge { from: *a, to: *b });
            }
            let outward = if let Some(last) = md_ids_only.last().copied() {
                vec![last]
            } else {
                branch_exits
            };
            self.header_exits.insert(hid, outward.clone());
            return (decision_id, outward);
        }

        let cluster_id = self.new_cluster_id();
        self.graph.clusters.push(Cluster {
            id: cluster_id,
            label: header.title.as_ref(),
            // TODO: clone() on copy
            parent: parent_cluster,
        });

        // Merge markdown children and direct nested roots, preserving each list's internal order
        let items_merged = merge_by_pos(&self.by_hid, &md_children, &nested_children);

        let mut first_rep: Option<_> = None;
        let mut prev_exits: Option<_> = None;
        for child_hid in items_merged.into_iter() {
            // If this child is a direct nested root for the current container, prebuild its scope
            // inside this container's cluster and use the scope's final exits.
            let mut prebuilt_scope_last_exits = None;
            if nested_children.contains(&child_hid) {
                let child_scope = self.by_hid[&child_hid].scope;
                self.build_scope_sequence(child_scope, visited_scopes, Some(cluster_id));
                let scope_items: Vec<Hid> = self
                    .index
                    .headers_in_scope_iter(child_scope)
                    .filter(|h| !self.has_md_parent.contains(&h.hid))
                    .map(|h| h.hid)
                    .collect();
                if let Some(&last_in_scope) = scope_items.last() {
                    if let Some(ex) = self.header_exits.get(&last_in_scope).cloned() {
                        prebuilt_scope_last_exits = Some(ex);
                    }
                }
            }

            let (entry, mut exits) = self.build_header(child_hid, visited_scopes, Some(cluster_id));
            if let Some(scope_exits) = prebuilt_scope_last_exits.take() {
                exits = scope_exits;
            }
            if first_rep.is_none() {
                first_rep = Some(entry);
            }
            if let Some(prev) = prev_exits.take() {
                for e in prev {
                    self.graph.edges.push(Edge { from: e, to: entry });
                }
            }
            prev_exits = Some(exits);
        }
        let entry = first_rep.unwrap_or_else(|| {
            // Create a placeholder node if the container is empty.
            let node_id = self.new_node_id();
            let span = SerializedSpan::serialize(&header.span);
            self.span_map.insert(node_id, span.clone());
            self.graph.nodes.push(Node {
                id: node_id,
                label: header.title.as_ref(),
                kind: NodeKind::Header(hid, Some(span)),
                cluster: Some(cluster_id),
            });
            node_id
        });
        let exits = prev_exits.unwrap_or_else(|| vec![entry]);
        self.header_entry.insert(hid, entry);
        self.header_exits.insert(hid, exits.clone());
        (entry, exits)
    }

    fn render_calls_for_header(
        &mut self,
        hid: Hid,
        header_rep_id: NodeId,
        cluster: Option<ClusterId>,
    ) {
        if let Some(callees) = self.index.header_calls.get(&hid) {
            for callee in callees {
                // TODO: get rid of `callee.clone()`
                if let Some(&cached_id) = self.call_node_cache.get(&(hid, callee)) {
                    self.graph.edges.push(Edge {
                        from: cached_id,
                        to: header_rep_id,
                    });
                    continue;
                }
                let call_node_id = self.new_node_id();
                self.graph.nodes.push(Node {
                    id: call_node_id,
                    label: callee.as_str(),
                    kind: NodeKind::Call {
                        header: hid,
                        callee,
                    },
                    cluster,
                });
                self.graph.edges.push(Edge {
                    from: call_node_id,
                    to: header_rep_id,
                });
                self.call_node_cache.insert((hid, callee), call_node_id);
            }
        }
    }

    fn new_node_id(&mut self) -> NodeId {
        let id = NodeId(self.next_node);
        self.next_node += 1;
        id
    }
    fn new_cluster_id(&mut self) -> ClusterId {
        let id = ClusterId(self.next_cluster);
        self.next_cluster += 1;
        id
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClusterId(u32);

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeId(u32);

impl serde::Serialize for NodeId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

impl std::fmt::Debug for ClusterId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}
impl std::fmt::Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n{}", self.0)
    }
}

impl std::fmt::Display for ClusterId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sg{}", self.0)
    }
}

fn build_scope_roots(index: &HeaderIndex) -> HashMap<ScopeId, Hid> {
    // iterate the headers by scope order. The first one to appear is the scope root.
    let mut scope_root = HashMap::new();
    for h in &index.headers {
        scope_root.entry(h.scope).or_insert(h.hid);
    }
    scope_root
}

fn all_nested_targets<'iter>(
    index: &'iter HeaderIndex,
    by_hid: &'iter HashMap<Hid, &'iter RenderableHeader>,
) -> impl Iterator<Item = Hid> + 'iter {
    nested_scope_edges(index, by_hid).map(|(_, c)| c)
}

/// Edges that cross a scope enter, i.e not just header changes.
fn nested_scope_edges<'iter>(
    index: &'iter HeaderIndex,
    by_hid: &'iter HashMap<Hid, &'iter RenderableHeader>,
) -> impl Iterator<Item = (Hid, Hid)> + 'iter {
    index
        .nested_edges_hid_iter()
        .filter(|(p, c)| by_hid[p].scope != by_hid[c].scope)
        .copied()
}

/// Compute a tuple position key for stable ordering comparisons
fn pos_tuple<'index>(
    by_hid: &HashMap<Hid, &'index RenderableHeader>,
    hid: Hid,
) -> (&'index Path, usize) {
    let h = by_hid[&hid];
    (h.span.file.path_buf().as_ref(), h.span.start)
}

/// Merge two already-ordered lists by source position, preserving internal order
fn merge_by_pos<'index, 'iter>(
    by_hid: &'iter HashMap<Hid, &'index RenderableHeader>,
    lhs: &'iter [Hid],
    rhs: &'iter [Hid],
) -> impl Iterator<Item = Hid> + 'iter
where
    'index: 'iter,
{
    return State { by_hid, lhs, rhs };

    // implement iterator manually for size hint + exact size.
    struct State<'index, 'iter> {
        by_hid: &'iter HashMap<Hid, &'index RenderableHeader>,
        lhs: &'iter [Hid],
        rhs: &'iter [Hid],
    }

    impl ExactSizeIterator for State<'_, '_> {
        fn len(&self) -> usize {
            self.lhs.len() + self.rhs.len()
        }
    }

    impl Iterator for State<'_, '_> {
        type Item = Hid;

        // exact sized
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.len();
            (len, Some(len))
        }

        fn next(&mut self) -> Option<Self::Item> {
            match (self.lhs, self.rhs) {
                ([l, lrest @ ..], [r, rrest @ ..]) => Some(
                    if pos_tuple(self.by_hid, *l) <= pos_tuple(self.by_hid, *r) {
                        self.lhs = lrest;
                        *l
                    } else {
                        self.rhs = rrest;
                        *r
                    },
                ),
                ([l, rest @ ..], []) => {
                    self.lhs = rest;
                    Some(*l)
                }
                ([], [r, rest @ ..]) => {
                    self.rhs = rest;
                    Some(*r)
                }
                ([], []) => None,
            }
        }
    }
}

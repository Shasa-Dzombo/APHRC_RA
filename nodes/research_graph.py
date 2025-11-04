"""
Research graph management and relationships between research components
"""
from typing import List, Dict, Optional, Set
from uuid import UUID
from enum import Enum
from pydantic import BaseModel
from datetime import datetime
from .research_nodes import BaseNode, NodeType

class RelationType(str, Enum):
    BASED_ON = "based_on"  # Topic/Dataset -> Research Questions
    SUPPORTS = "supports"  # Literature -> Research Questions/Findings
    ANALYZES = "analyzes"  # Analysis -> Dataset
    PRODUCES = "produces"  # Analysis -> Findings
    RELATES_TO = "relates_to"  # General relationship
    CITES = "cites"  # Literature -> Literature
    EXTENDS = "extends"  # Research Question -> Research Question
    CONTRADICTS = "contradicts"  # Finding -> Finding or Literature -> Literature

class Relationship(BaseModel):
    source_id: UUID
    target_id: UUID
    relation_type: RelationType
    strength: float = 1.0  # Relationship strength/confidence (0-1)
    metadata: Dict = {}
    created_at: datetime = datetime.utcnow()

class ResearchGraph:
    def __init__(self):
        self.nodes: Dict[UUID, BaseNode] = {}
        self.relationships: List[Relationship] = []
        self._adjacency_list: Dict[UUID, Set[UUID]] = {}

    def add_node(self, node: BaseNode) -> UUID:
        """Add a node to the research graph"""
        self.nodes[node.id] = node
        self._adjacency_list[node.id] = set()
        return node.id

    def add_relationship(self, relationship: Relationship):
        """Add a relationship between nodes"""
        if relationship.source_id not in self.nodes or relationship.target_id not in self.nodes:
            raise ValueError("Both source and target nodes must exist in the graph")
        
        self.relationships.append(relationship)
        self._adjacency_list[relationship.source_id].add(relationship.target_id)

    def get_related_nodes(self, node_id: UUID, relation_type: Optional[RelationType] = None) -> List[BaseNode]:
        """Get all nodes related to a given node, optionally filtered by relationship type"""
        related_nodes = []
        for rel in self.relationships:
            if rel.source_id == node_id and (relation_type is None or rel.relation_type == relation_type):
                related_nodes.append(self.nodes[rel.target_id])
        return related_nodes

    def get_node_chain(self, start_node_id: UUID, relation_types: List[RelationType]) -> List[List[BaseNode]]:
        """Get chains of nodes following specified relationship patterns"""
        def dfs(current_id: UUID, target_relations: List[RelationType], path: List[BaseNode]) -> List[List[BaseNode]]:
            if not target_relations:
                return [path]
            
            results = []
            current_relation = target_relations[0]
            
            for rel in self.relationships:
                if rel.source_id == current_id and rel.relation_type == current_relation:
                    target_node = self.nodes[rel.target_id]
                    new_results = dfs(rel.target_id, target_relations[1:], path + [target_node])
                    results.extend(new_results)
            
            return results

        start_node = self.nodes[start_node_id]
        return dfs(start_node_id, relation_types, [start_node])

    def find_paths(self, start_id: UUID, end_id: UUID, max_depth: int = 5) -> List[List[BaseNode]]:
        """Find all paths between two nodes up to a maximum depth"""
        def dfs(current_id: UUID, target_id: UUID, depth: int, path: List[BaseNode], visited: Set[UUID]) -> List[List[BaseNode]]:
            if depth < 0 or current_id in visited:
                return []
            if current_id == target_id:
                return [path]
            
            visited.add(current_id)
            paths = []
            
            for next_id in self._adjacency_list[current_id]:
                next_node = self.nodes[next_id]
                new_paths = dfs(next_id, target_id, depth - 1, path + [next_node], visited.copy())
                paths.extend(new_paths)
            
            return paths

        start_node = self.nodes[start_id]
        return dfs(start_id, end_id, max_depth, [start_node], set())
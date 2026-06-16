import networkx as nx
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple

class BiomedicalKnowledgeGraph:
    """
    Manages the NetworkX relational knowledge graph.
    Links genomic variations through cellular pathway nodes to target therapies.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_from_mutation_data(self, mut_name: str, data: Dict[str, Any]):
        """
        Dynamically populates the directed graph based on the fetched mutation records.
        """
        gene_name = data.get("gene", "UnknownGene")
        protein_impact = data.get("protein_impact", {})
        uniprot_id = protein_impact.get("uniprot_id", "UnknownProt")
        
        # 1. Add core entities as nodes
        self.graph.add_node(mut_name, type="Mutation", label=mut_name, color="#E11D48") # Rose
        self.graph.add_node(gene_name, type="Gene", label=gene_name, color="#EA580C") # Orange
        self.graph.add_node(uniprot_id, type="Protein", label=f"Protein ({uniprot_id})", color="#D97706") # Amber
        
        # Add edges
        self.graph.add_edge(mut_name, gene_name, relation="affects")
        self.graph.add_edge(gene_name, uniprot_id, relation="encodes")
        
        # 2. Add pathways
        for pathway in data.get("pathways", []):
            p_id = pathway.get("id", "UnknownPath")
            p_name = pathway.get("name", "Unknown Pathway")
            self.graph.add_node(p_id, type="Pathway", label=p_name, color="#0D9488") # Teal
            self.graph.add_edge(uniprot_id, p_id, relation="participates_in")
            
            # Map default disease link from pathway if oncology related
            disease_name = "Oncology Disease Complex"
            if "homologous" in p_name.lower() or "brca" in mut_name.lower():
                disease_name = "Breast / Ovarian Cancer"
            elif "egfr" in mut_name.lower():
                disease_name = "Non-Small Cell Lung Cancer (NSCLC)"
            elif "braf" in mut_name.lower():
                disease_name = "Malignant Melanoma"
            elif "kras" in mut_name.lower():
                disease_name = "Colorectal / Lung Adenocarcinoma"
            elif "alk" in mut_name.lower():
                disease_name = "Neuroblastoma / NSCLC"
                
            self.graph.add_node(disease_name, type="Disease", label=disease_name, color="#4F46E5") # Indigo
            self.graph.add_edge(p_id, disease_name, relation="associated_with")
            
        # 3. Add Drugs/Therapies
        for drug in data.get("drugs", []):
            drug_name = drug.get("name", "Unknown Drug")
            drug_type = drug.get("type", "inhibitor")
            self.graph.add_node(drug_name, type="Drug", label=f"{drug_name} ({drug_type})", color="#0891B2") # Cyan
            self.graph.add_edge(drug_name, uniprot_id, relation="targets")

    def get_traversal_path(self, mut_name: str) -> List[str]:
        """
        Finds a logical path tracing from Mutation → Gene → Protein → Pathway → Disease.
        """
        path = []
        if mut_name not in self.graph:
            return path
            
        current = mut_name
        path.append(current)
        
        # Greedy traversal of children based on entity types
        visited = {current}
        while True:
            successors = list(self.graph.successors(current))
            if not successors:
                break
            
            # Filter unvisited successors
            unvisited = [s for s in successors if s not in visited]
            if not unvisited:
                break
                
            # Select the next logical step in our chain
            next_node = unvisited[0]
            current = next_node
            path.append(current)
            visited.add(current)
            
        return path

    def generate_plotly_figure(self) -> go.Figure:
        """
        Computes a spring-layout and returns a highly aesthetic Plotly scatter graph
        to render natively inside Streamlit.
        """
        if len(self.graph) == 0:
            # Empty placeholder figure
            fig = go.Figure()
            fig.update_layout(title="No nodes in Knowledge Graph")
            return fig

        # Compute layouts
        pos = nx.spring_layout(self.graph, k=1.0, iterations=50, seed=42)
        
        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#475569'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            attrs = self.graph.nodes[node]
            n_type = attrs.get("type", "Unknown")
            n_label = attrs.get("label", node)
            color = attrs.get("color", "#64748B")
            
            node_colors.append(color)
            node_text.append(f"<b>Type:</b> {n_type}<br><b>Entity:</b> {n_label}")
            
            # Scale sizes by type
            size_map = {"Mutation": 24, "Gene": 20, "Protein": 20, "Pathway": 18, "Disease": 22, "Drug": 22}
            node_sizes.append(size_map.get(n_type, 16))
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[self.graph.nodes[node].get("label", node).split(" (")[0] for node in self.graph.nodes()],
            textposition="top center",
            textfont=dict(size=10, color='#E2E8F0'),
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='#1E293B')
            )
        )
        
        # Add hover text metadata
        node_trace.hovertext = node_text
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=10, l=10, r=10, t=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                dragmode='pan'
            )
        )
        
        return fig

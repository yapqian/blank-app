import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class MultiRelationalGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, article_in_channels, source_in_channels):
        super().__init__()
        # Matches your training: (source_in, article_in)
        self.conv1_pb = SAGEConv((source_in_channels, article_in_channels), hidden_channels)
        self.conv1_sd = SAGEConv(article_in_channels, hidden_channels)
        self.conv2_pb = SAGEConv((source_in_channels, hidden_channels), out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Relationship: Article -> Published By -> Source
        pb_edge_index = edge_index_dict[('article', 'published_by', 'source')]
        source_to_article_edge_index = pb_edge_index.flip([0])

        x_article_from_source = self.conv1_pb(
            (x_dict['source'], x_dict['article']),
            source_to_article_edge_index
        ).relu()

        # Relationship: Article -> Same Day -> Article
        if ('article', 'same_day', 'article') in edge_index_dict:
            x_article_same_day = self.conv1_sd(
                x_dict['article'],
                edge_index_dict[('article', 'same_day', 'article')]
            ).relu()
        else:
            x_article_same_day = torch.zeros_like(x_article_from_source)

        x_article_updated = x_article_from_source + x_article_same_day

        out_article = self.conv2_pb(
            (x_dict['source'], x_article_updated),
            source_to_article_edge_index
        )

        return {'article': out_article}
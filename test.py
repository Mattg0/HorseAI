from utils.env_setup import AppConfig
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator



def main():
    orchestrator= FeatureEmbeddingOrchestrator()
    return orchestrator

if __name__ == '__main__':
    main()
import argparse
from app.ingest import ingest

parser = argparse.ArgumentParser(description="Index the Constitution of India PDF")
parser.add_argument(
    "--pdf",
    default="data/constitution_of_india.pdf",
    help="Path to the Constitution PDF",
)
args = parser.parse_args()

count = ingest(args.pdf)
print(f"Indexed {count} chunks into ChromaDB.")

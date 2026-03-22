import argparse
import csv
import json
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Chunk CSV rows from stdin using RecursiveCharacterTextSplitter.'
    )
    parser.add_argument(
        '--column', type=str, default='text', help='CSV column to chunk (default: text)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=512,
        help='Max chunk size in characters (default: 512)',
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='Overlap between chunks (default: 50)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='chunks.json',
        help='Output file path (default: chunks.json)',
    )
    args = parser.parse_args()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    reader = csv.DictReader(sys.stdin)
    if args.column not in (reader.fieldnames or []):
        print(f"Error: column '{args.column}' not found in CSV.", file=sys.stderr)
        sys.exit(1)

    chunks: list[str] = []
    for row in reader:
        text = row[args.column].strip()
        if text:
            chunks.extend(splitter.split_text(text))

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f'Written {len(chunks)} chunks to {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()

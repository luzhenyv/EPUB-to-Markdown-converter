# EPUB to Markdown/Text Converter

A command-line tool to convert EPUB files to either Markdown or plain text format, with optional support for image extraction and content optimization using Ollama.

## Features

- Supports EPUB2 and EPUB3 formats
- Extracts and preserves document structure
- Handles images and other media files
- Converts HTML content to either plain text or Markdown
- Optional LLM-based conversion using Ollama (gemma3:27b)
- Provides detailed logging and error handling

## Requirements

- Python 3.8 or higher
- Required Python packages (see requirements.txt)
- Ollama (only required if using LLM conversion)

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. If you plan to use LLM conversion, make sure Ollama is running with the desired model:
   ```bash
   ollama pull gemma3:27b  # or your preferred model
   ```

## Usage

```bash
python extract_epub_to_markdown.py <epub_file> <output_directory> [options]
```

### Basic Usage (Plain Text Conversion)
```bash
python extract_epub_to_markdown.py "book.epub" ./output
```

### LLM-based Markdown Conversion
```bash
python extract_epub_to_markdown.py "book.epub" ./output --use-llm
```

### Full Options
```bash
python extract_epub_to_markdown.py "book.epub" ./output \
    --use-llm \
    --model gemma3:27b \
    --temperature 0.0 \
    --title "Book Title" \
    --author "Author Name" \
    --publisher "Publisher" \
    --isbn "1234567890" \
    --publication-date "2024" \
    --description "Book description"
```

a demo case looks like that:

```bash
python extract_epub_to_markdown.py "Impro-Improvisation and the Theatre.epub" ./output \
    --use-llm \
    --model gemma3:27b-it-qat  \
    --title "Impro: Improvisation and the Theatre" \
    --author "Keith Johnstone" \
    --publisher "Routledge" \
    --isbn "9780878301171" \
    --publication-date "1987" \
    --description "Keith Johnstone's involvement with the theatre began when George Devine and Tony Richardson, artistic directors of the Royal Court Theatre, commissioned a play from him. This was in 1956. A few years later he was himself Associate Artistic Director, working as a play-reader and director, in particular helping to run the Writers' Group. The improvisatory techniques and exercises evolved there to foster spontaneity and narrative skills were developed further in the actors' studio then in demonstrations to schools and colleges and ultimately in the founding of a company of performers, called The Theatre Machine. Divided into four sections, 'Status', 'Spontaneity', 'Narrative Skills', and 'Masks and Trance', arranged more or less in the order a group might approach them, the book sets out the specific techniques and exercises which Johnstone has himself found most useful and most stimulating. The result is both an ideas book and a fascinating exploration of the nature of spontaneous creativity."


```

### Conversion Options

- `--use-llm`: Enable LLM-based conversion to Markdown (default: False)
- `--model`: Specify the Ollama model to use (default: gemma3:27b)
- `--temperature`: Set the temperature for LLM generation (default: 0.0)

### Book Information Parameters (Optional)

- `--title`: Book title
- `--author`: Book author
- `--publisher`: Book publisher
- `--isbn`: Book ISBN
- `--publication-date`: Publication date
- `--description`: Book description

These parameters are optional but recommended when using LLM conversion as they help improve the conversion quality by providing context.

## Output Structure

```
output_directory/
├── pic/
│   └── [extracted images]
├── chapter1.txt  # or .md if using LLM
├── chapter2.txt  # or .md if using LLM
└── conversion.log
```

## Error Handling

The tool provides detailed error messages and logging:
- Encrypted EPUB files are detected and reported
- Failed image extractions are logged
- Conversion errors are recorded in the log file
- LLM optimization failures are handled gracefully

## License

MIT License 
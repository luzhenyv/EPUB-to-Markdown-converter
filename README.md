# EPUB to Markdown Converter

A command-line tool to convert EPUB files to Markdown format, with support for image extraction and content optimization using Ollama.

## Features

- Supports EPUB2 and EPUB3 formats
- Extracts and preserves document structure
- Handles images and other media files
- Converts HTML content to Markdown
- Uses Ollama (gemma3:27b) for content optimization
- Provides detailed logging and error handling

## Requirements

- Python 3.8 or higher
- Ollama running locally with gemma3:27b model
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure Ollama is running with the gemma3:27b model:
   ```bash
   ollama pull gemma3:27b
   ```

## Usage

```bash
python extract_epub_to_markdown.py <epub_file> <output_directory>
```

Example:
```bash
python extract_epub_to_markdown.py book.epub ./output
```

The tool will:
1. Create the output directory if it doesn't exist
2. Create a 'pic' subdirectory for images
3. Convert the EPUB file to Markdown
4. Generate a conversion.log file with detailed information

## Output Structure

```
output_directory/
├── pic/
│   └── [extracted images]
├── chapter1.md
├── chapter2.md
└── conversion.log
```

## Error Handling

The tool provides detailed error messages and logging:
- Encrypted EPUB files are detected and reported
- Failed image extractions are logged
- Conversion errors are recorded in the log file
- Ollama optimization failures are handled gracefully

## License

MIT License 
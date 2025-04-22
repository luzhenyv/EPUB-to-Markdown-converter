#!/usr/bin/env python3
import os
import sys
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import zipfile
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import ollama
from tqdm import tqdm
import argparse

# System content for Ollama optimization
OLLAMA_SYSTEM_PROMPT = """You are a professional EPUB to Markdown converter. You MUST:
1. Convert HTML to Markdown while preserving ALL original content exactly
2. Apply proper Markdown formatting:
   - Headings: <h1>/<class="h1"> → #, <h2>/<class="h2"> → ##, etc.
   - Paragraphs: <p> → proper paragraph breaks (double newline)
   - Lists: <ul>/<ol> → Markdown lists
   - Blockquotes: <blockquote> → >
   - Tables: <table> → Markdown table syntax
   - Links: <a href="...">text</a> → [text](...)
   - Emphasis: <i>/<em> → *text*, <b>/<strong> → **text**
   - Line breaks: <br> → single newline, <hr> → ---
   - Special headings: Convert classes like "title", "chapter", "section" appropriately
3. Remove ALL HTML attributes and tags while keeping the text content intact
4. Do NOT interpret, change, add to, or remove any content
5. Return ONLY the formatted Markdown content
6. CRITICAL: Do NOT add any notes, explanations, commentary, or descriptions about your conversion process
7. NEVER include sections like "Notes on the conversion" or similar explanations
8. Book context (for reference only): {book_info}"""

# User prompt for HTML to Markdown conversion
OLLAMA_USER_PROMPT = """Convert this HTML content to clean Markdown. 
Output ONLY the converted content with no explanations or notes of any kind.

HTML CONTENT:
--------------------------------
{content}
--------------------------------
"""

class EPUBExtractor:
    """Class responsible for extracting content from EPUB files."""
    
    def __init__(self, epub_path: str):
        """
        Initialize the EPUB extractor.
        
        Args:
            epub_path (str): Path to the EPUB file
        """
        self.epub_path = Path(epub_path)
        
    def check_encryption(self) -> bool:
        """
        Check if the EPUB file is encrypted.
        
        Returns:
            bool: True if encrypted, False otherwise
        """
        try:
            with zipfile.ZipFile(self.epub_path, "r") as epub:
                return "META-INF/encryption.xml" in epub.namelist()
        except Exception as e:
            logging.error(f"Error checking encryption: {e}")
            return True
            
    def extract_structure(self) -> Dict[str, Any]:
        """
        Extract the document structure from the EPUB file.
        
        Returns:
            Dict: Document structure tree
        """
        try:
            with zipfile.ZipFile(self.epub_path, "r") as epub:
                # Extract container.xml to find content.opf
                container_data = epub.read("META-INF/container.xml")
                container_root = ET.fromstring(container_data)
                content_path = container_root.find(".//{*}rootfile").get("full-path")

                # Parse content.opf
                content_data = epub.read(content_path)
                content_root = ET.fromstring(content_data)

                # Extract manifest and spine
                manifest = {}
                for item in content_root.findall(".//{*}manifest/{*}item"):
                    manifest[item.get("id")] = item.get("href")

                spine = []
                for itemref in content_root.findall(".//{*}spine/{*}itemref"):
                    spine.append(itemref.get("idref"))

                return {
                    "manifest": manifest,
                    "spine": spine,
                    "content_path": content_path,
                }
        except Exception as e:
            logging.error(f"Error extracting structure: {e}")
            return {}
            
    def read_chapter_content(self, chapter_id: str, manifest: Dict[str, str]) -> Optional[str]:
        """
        Read content of a specific chapter.
        
        Args:
            chapter_id (str): ID of the chapter to read
            manifest (Dict[str, str]): Manifest mapping IDs to file paths
            
        Returns:
            Optional[str]: Chapter content as string, or None if error
        """
        try:
            with zipfile.ZipFile(self.epub_path, "r") as epub:
                content_path = manifest[chapter_id]
                return epub.read(content_path).decode("utf-8")
        except Exception as e:
            logging.error(f"Error reading chapter {chapter_id}: {e}")
            return None
            
    def extract_image(self, src: str) -> Optional[bytes]:
        """
        Extract image data from EPUB.
        
        Args:
            src (str): Source path of the image
            
        Returns:
            Optional[bytes]: Image data, or None if error
        """
        try:
            with zipfile.ZipFile(self.epub_path, "r") as epub:
                return epub.read(src)
        except Exception as e:
            logging.error(f"Error extracting image {src}: {e}")
            return None


class ImageProcessor:
    """Class responsible for processing images from EPUB content."""
    
    def __init__(self, pic_dir: Path, epub_extractor: EPUBExtractor):
        """
        Initialize the image processor.
        
        Args:
            pic_dir (Path): Directory to save images
            epub_extractor (EPUBExtractor): EPUB extractor instance
        """
        self.pic_dir = pic_dir
        self.epub_extractor = epub_extractor
        self.pic_dir.mkdir(exist_ok=True)
        
    def process_images(self, content: str, chapter_id: str) -> Tuple[str, List[str]]:
        """
        Process and extract images from content.
        
        Args:
            content (str): HTML content
            chapter_id (str): Chapter identifier
            
        Returns:
            Tuple[str, List[str]]: Processed content and list of image paths
        """
        soup = BeautifulSoup(content, "html.parser")
        image_paths = []

        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue
                
            # Create descriptive name from alt text or src
            alt_text = img.get("alt", "")
            desc_name = alt_text.replace(" ", "_") if alt_text else Path(src).stem
            new_filename = f"{chapter_id}_{desc_name}_{len(image_paths)}{Path(src).suffix}"
            new_path = self.pic_dir / new_filename

            # Extract and save image
            img_data = self.epub_extractor.extract_image(src)
            if img_data:
                try:
                    with open(new_path, "wb") as f:
                        f.write(img_data)
                    # Update image reference in content
                    img["src"] = f"pic/{new_filename}"
                    image_paths.append(str(new_path))
                except Exception as e:
                    logging.warning(f"Failed to save image {src}: {e}")

        return str(soup), image_paths


class ContentConverter:
    """Class responsible for converting HTML content to Markdown or plain text."""
    
    def convert_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML content to Markdown format.
        
        Args:
            html_content (str): HTML content to convert
            
        Returns:
            str: Markdown content
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Convert headings
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(h.name[1])
            h.replace_with(f"{'#' * level} {h.get_text()}\n\n")

        # Convert paragraphs
        for p in soup.find_all("p"):
            p.replace_with(f"{p.get_text()}\n\n")

        # Convert lists
        for ul in soup.find_all("ul"):
            items = [f"- {li.get_text()}" for li in ul.find_all("li")]
            ul.replace_with("\n".join(items) + "\n\n")

        for ol in soup.find_all("ol"):
            items = [f"{i+1}. {li.get_text()}" for i, li in enumerate(ol.find_all("li"))]
            ol.replace_with("\n".join(items) + "\n\n")

        # Convert blockquotes
        for blockquote in soup.find_all("blockquote"):
            text = blockquote.get_text().strip()
            blockquote.replace_with(f"> {text}\n\n")

        return str(soup)

    def convert_to_plain_text(self, html_content: str) -> str:
        """
        Convert HTML content to plain text format.
        
        Args:
            html_content (str): HTML content to convert
            
        Returns:
            str: Plain text content
        """
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text


class LLMOptimizer:
    """Class responsible for optimizing content using LLM."""
    
    def __init__(self, model_name: str, book_info: Dict[str, Any], temperature: float = 0.0):
        """
        Initialize the LLM optimizer.
        
        Args:
            model_name (str): Name of the Ollama model to use
            book_info (Dict[str, Any]): Dictionary containing book information
            temperature (float): Temperature for LLM generation
        """
        self.model_name = model_name
        self.book_info = book_info
        self.temperature = temperature
        
    def optimize_content(self, markdown_content: str, max_tokens: int = 2000) -> str:
        """
        Use Ollama to optimize the Markdown content in chunks.
        
        Args:
            markdown_content (str): Markdown content to optimize
            max_tokens (int): Maximum number of tokens per chunk
            
        Returns:
            str: Optimized Markdown content
        """
        try:
            # Split content into paragraphs
            paragraphs = markdown_content.split('\n')
            optimized_content = []
            current_chunk = []
            current_length = 0

            for para in paragraphs:
                # Estimate token count (roughly 1 token per 4 characters)
                para_tokens = len(para) // 4
                
                # If adding this paragraph would exceed max_tokens, process current chunk
                if current_length + para_tokens > max_tokens and current_chunk:
                    # Process current chunk
                    chunk_content = '\n'.join(current_chunk)
                    response = self._call_ollama_chat(chunk_content)
                    optimized_content.append(response["message"]["content"])
                    
                    # Reset for next chunk
                    current_chunk = []
                    current_length = 0
                
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_length += para_tokens

            # Process any remaining content
            if current_chunk:
                chunk_content = "\n".join(current_chunk)
                response = self._call_ollama_chat(chunk_content)
                optimized_content.append(response["message"]["content"])

            # Join all optimized chunks
            return '\n'.join(optimized_content)

        except Exception as e:
            logging.warning(f"Ollama optimization failed: {e}")
            return markdown_content
            
    def _call_ollama_chat(self, content: str) -> Dict[str, Any]:
        """
        Make a call to Ollama chat API.
        
        Args:
            content (str): Content to be processed
            
        Returns:
            Dict[str, Any]: Response from Ollama API
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": OLLAMA_SYSTEM_PROMPT.format(book_info=self.book_info),
                    },
                    {"role": "user", "content": OLLAMA_USER_PROMPT.format(content=content)},
                ],
                options={"temperature": self.temperature}
            )
            return response
        except Exception as e:
            logging.warning(f"Ollama API call failed: {e}")
            return {"message": {"content": content}}


class EPUBConverter:
    """Main class coordinating the EPUB conversion process."""
    
    def __init__(self, epub_path: str, output_dir: str, book_info: Dict[str, Any] = None, 
                 model_name: str = "gemma3:27b", use_llm: bool = False, temperature: float = 0.0):
        """
        Initialize the EPUB converter.
        
        Args:
            epub_path (str): Path to the EPUB file
            output_dir (str): Directory to save the converted files
            book_info (Dict[str, Any]): Dictionary containing book information
            model_name (str): Name of the Ollama model to use
            use_llm (bool): Whether to use LLM for conversion
            temperature (float): Temperature for LLM generation
        """
        self.output_dir = Path(output_dir)
        self.pic_dir = self.output_dir / "pic"
        self.log_file = self.output_dir / "conversion.log"
        self.book_info = book_info or {}
        self.use_llm = use_llm
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.log_file), logging.StreamHandler()],
        )
        
        # Initialize components
        self.epub_extractor = EPUBExtractor(epub_path)
        self.image_processor = ImageProcessor(self.pic_dir, self.epub_extractor)
        self.content_converter = ContentConverter()
        self.llm_optimizer = LLMOptimizer(model_name, self.book_info, temperature)
        
    def convert(self) -> bool:
        """
        Main conversion process.
        
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        try:
            # Check if file is encrypted
            if self.epub_extractor.check_encryption():
                logging.error("EPUB file is encrypted. Conversion aborted.")
                return False

            # Extract structure
            structure = self.epub_extractor.extract_structure()
            if not structure:
                logging.error("Failed to extract EPUB structure")
                return False

            # Process each chapter
            for chapter_id in tqdm(structure["spine"], desc="Converting chapters"):
                self._process_chapter(chapter_id, structure["manifest"])

            logging.info("Conversion completed successfully")
            return True

        except Exception as e:
            logging.error(f"Conversion failed: {e}")
            return False
            
    def _process_chapter(self, chapter_id: str, manifest: Dict[str, str]) -> None:
        """
        Process a single chapter from the EPUB.
        
        Args:
            chapter_id (str): ID of the chapter to process
            manifest (Dict[str, str]): Manifest mapping IDs to file paths
        """
        # Read chapter content
        content = self.epub_extractor.read_chapter_content(chapter_id, manifest)
        if not content:
            return
            
        # Process images
        content, image_paths = self.image_processor.process_images(content, chapter_id)
        
        # Convert content based on mode
        if self.use_llm:
            # Use LLM for conversion to Markdown
            optimized_content = self.llm_optimizer.optimize_content(content)
            file_extension = "md"
        else:
            # Convert to plain text
            optimized_content = self.content_converter.convert_to_plain_text(content)
            file_extension = "txt"
            
        # Save chapter
        title = chapter_id.replace('_', ' ').title()
        output_path = self.output_dir / f"# {title}.{file_extension}"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(optimized_content)
        except Exception as e:
            logging.error(f"Error saving chapter {chapter_id}: {e}")


def main():
    """Parse command line arguments and run the converter."""
    parser = argparse.ArgumentParser(description='Convert EPUB to Markdown or plain text')
    parser.add_argument('epub_file', help='Path to the EPUB file')
    parser.add_argument('output_directory', help='Directory to save the converted files')
    parser.add_argument('--title', help='Book title')
    parser.add_argument('--author', help='Book author')
    parser.add_argument('--publisher', help='Book publisher')
    parser.add_argument('--isbn', help='Book ISBN')
    parser.add_argument('--publication-date', help='Publication date')
    parser.add_argument('--description', help='Book description')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for conversion to Markdown (default: False)')
    parser.add_argument('--model', default='gemma3:27b', help='Ollama model to use (default: gemma3:27b)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for LLM generation (default: 0.0)')

    args = parser.parse_args()

    # Create book info dictionary
    book_info = {
        'title': args.title,
        'author': args.author,
        'publisher': args.publisher,
        'isbn': args.isbn,
        'publication_date': args.publication_date,
        'description': args.description
    }

    # Filter out None values
    book_info = {k: v for k, v in book_info.items() if v is not None}

    converter = EPUBConverter(
        args.epub_file,
        args.output_directory,
        book_info,
        model_name=args.model,
        use_llm=args.use_llm,
        temperature=args.temperature
    )
    success = converter.convert()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

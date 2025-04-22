#!/usr/bin/env python3
import os
import sys
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import ollama
from tqdm import tqdm

class EPUBConverter:
    def __init__(self, epub_path: str, output_dir: str):
        """
        Initialize the EPUB converter.
        
        Args:
            epub_path (str): Path to the EPUB file
            output_dir (str): Directory to save the converted files
        """
        self.epub_path = Path(epub_path)
        self.output_dir = Path(output_dir)
        self.pic_dir = self.output_dir / "pic"
        self.log_file = self.output_dir / "conversion.log"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        # Initialize data structures
        self.structure_tree = {}
        self.image_map = {}
        self.complex_tables = []
        
    def check_encryption(self) -> bool:
        """
        Check if the EPUB file is encrypted.
        
        Returns:
            bool: True if encrypted, False otherwise
        """
        try:
            with zipfile.ZipFile(self.epub_path, 'r') as epub:
                return 'META-INF/encryption.xml' in epub.namelist()
        except Exception as e:
            logging.error(f"Error checking encryption: {e}")
            return True
            
    def extract_structure(self) -> Dict:
        """
        Extract the document structure from the EPUB file.
        
        Returns:
            Dict: Document structure tree
        """
        try:
            with zipfile.ZipFile(self.epub_path, 'r') as epub:
                # Extract container.xml to find content.opf
                container_data = epub.read('META-INF/container.xml')
                container_root = ET.fromstring(container_data)
                content_path = container_root.find('.//{*}rootfile').get('full-path')
                
                # Parse content.opf
                content_data = epub.read(content_path)
                content_root = ET.fromstring(content_data)
                
                # Extract manifest and spine
                manifest = {}
                for item in content_root.findall('.//{*}manifest/{*}item'):
                    manifest[item.get('id')] = item.get('href')
                
                spine = []
                for itemref in content_root.findall('.//{*}spine/{*}itemref'):
                    spine.append(itemref.get('idref'))
                
                return {
                    'manifest': manifest,
                    'spine': spine,
                    'content_path': content_path
                }
        except Exception as e:
            logging.error(f"Error extracting structure: {e}")
            return {}
            
    def process_images(self, content: str, chapter_id: str) -> Tuple[str, List[str]]:
        """
        Process and extract images from content.
        
        Args:
            content (str): HTML content
            chapter_id (str): Chapter identifier
            
        Returns:
            Tuple[str, List[str]]: Processed content and list of image paths
        """
        soup = BeautifulSoup(content, 'html.parser')
        image_paths = []
        
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                # Create descriptive name from alt text or src
                alt_text = img.get('alt', '')
                desc_name = alt_text.replace(' ', '_') if alt_text else Path(src).stem
                new_filename = f"{chapter_id}_{desc_name}_{len(image_paths)}{Path(src).suffix}"
                new_path = self.pic_dir / new_filename
                
                # Extract and save image
                try:
                    with zipfile.ZipFile(self.epub_path, 'r') as epub:
                        img_data = epub.read(src)
                        with open(new_path, 'wb') as f:
                            f.write(img_data)
                except Exception as e:
                    logging.warning(f"Failed to extract image {src}: {e}")
                    continue
                
                # Update image reference in content
                img['src'] = f"pic/{new_filename}"
                image_paths.append(new_path)
        
        return str(soup), image_paths
        
    def convert_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML content to Markdown format.
        
        Args:
            html_content (str): HTML content to convert
            
        Returns:
            str: Markdown content
        """
        # Basic HTML to Markdown conversion
        # This is a simplified version - you might want to use a library like html2text
        # or implement more sophisticated conversion rules
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Convert headings
        for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(h.name[1])
            h.replace_with(f"{'#' * level} {h.get_text()}\n\n")
        
        # Convert paragraphs
        for p in soup.find_all('p'):
            p.replace_with(f"{p.get_text()}\n\n")
        
        # Convert lists
        for ul in soup.find_all('ul'):
            items = [f"- {li.get_text()}" for li in ul.find_all('li')]
            ul.replace_with('\n'.join(items) + '\n\n')
            
        for ol in soup.find_all('ol'):
            items = [f"{i+1}. {li.get_text()}" for i, li in enumerate(ol.find_all('li'))]
            ol.replace_with('\n'.join(items) + '\n\n')
        
        # Convert blockquotes
        for blockquote in soup.find_all('blockquote'):
            text = blockquote.get_text().strip()
            blockquote.replace_with(f"> {text}\n\n")
        
        return str(soup)
        
    def optimize_with_ollama(self, markdown_content: str) -> str:
        """
        Use Ollama to optimize the Markdown content.
        
        Args:
            markdown_content (str): Markdown content to optimize
            
        Returns:
            str: Optimized Markdown content
        """
        try:
            response = ollama.chat(
                model='gemma3:27b',
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a Markdown formatting expert. Please optimize the following Markdown content while preserving its meaning and structure.'
                    },
                    {
                        'role': 'user',
                        'content': markdown_content
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            logging.warning(f"Ollama optimization failed: {e}")
            return markdown_content
            
    def convert(self) -> bool:
        """
        Main conversion process.
        
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        try:
            # Check if file is encrypted
            if self.check_encryption():
                logging.error("EPUB file is encrypted. Conversion aborted.")
                return False
                
            # Create output directories
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.pic_dir.mkdir(exist_ok=True)
            
            # Extract structure
            structure = self.extract_structure()
            if not structure:
                logging.error("Failed to extract EPUB structure")
                return False
                
            # Process each chapter
            for chapter_id in tqdm(structure['spine'], desc="Converting chapters"):
                try:
                    with zipfile.ZipFile(self.epub_path, 'r') as epub:
                        content_path = structure['manifest'][chapter_id]
                        content = epub.read(content_path).decode('utf-8')
                        
                        # Process images
                        content, image_paths = self.process_images(content, chapter_id)
                        
                        # Convert to Markdown
                        markdown_content = self.convert_to_markdown(content)
                        
                        # Optimize with Ollama
                        optimized_content = self.optimize_with_ollama(markdown_content)
                        
                        # Save chapter
                        output_path = self.output_dir / f"{chapter_id}.md"
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(optimized_content)
                            
                except Exception as e:
                    logging.error(f"Error processing chapter {chapter_id}: {e}")
                    continue
                    
            logging.info("Conversion completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Conversion failed: {e}")
            return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_epub_to_markdown.py <epub_file> <output_directory>")
        sys.exit(1)
        
    epub_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    converter = EPUBConverter(epub_path, output_dir)
    success = converter.convert()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()

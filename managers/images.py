"""
Image management for Publicia
"""
import base64
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional

logger = logging.getLogger(__name__)

class ImageManager:
    """Manages image storage, descriptions, and retrieval."""
    
    def __init__(self, base_dir: str = "images", document_manager = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Reference to DocumentManager
        self.document_manager = document_manager
        
        # Storage for image metadata
        self.metadata = {}
        
        # Load existing images
        self._load_images()

    async def add_image(self, name: str, image_data: bytes, description: str = None): # Changed to async def
        """Add a new image to the system with optional description."""
        try:
            # Generate a unique ID for the image
            image_id = str(uuid.uuid4())
            
            # Use .png for all images for simplicity
            image_path = self.base_dir / f"{image_id}.png"
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # Create metadata
            self.metadata[image_id] = {
                'name': name,
                'added': datetime.now().isoformat(),
                'description': description,
                'path': str(image_path)
            }
            
            # Save metadata
            self._save_metadata()
            
            # Add description as a document for search if provided
            if description and self.document_manager:
                doc_name = f"image_{image_id}.txt"
                await self.document_manager.add_document(doc_name, description) # Added await
                
                # Add image reference to document metadata
                if doc_name in self.document_manager.metadata:
                    self.document_manager.metadata[doc_name]['image_id'] = image_id
                    self.document_manager.metadata[doc_name]['image_name'] = name
                    self.document_manager._save_to_disk()
            
            return image_id
            
        except Exception as e:
            logger.error(f"Error adding image {name}: {e}")
            raise
    
    def get_image(self, image_id: str) -> Tuple[bytes, str]:
        """Get image data and description by ID."""
        if image_id not in self.metadata:
            raise ValueError(f"Image ID {image_id} not found")
        
        image_path = Path(self.metadata[image_id]['path'])
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} not found")
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        description = self.metadata[image_id].get('description', '')
        
        return image_data, description
    
    def get_base64_image(self, image_id: str) -> str:
        """Get base64-encoded image data by ID."""
        if image_id not in self.metadata:
            raise ValueError(f"Image ID {image_id} not found")
            
        image_path = Path(self.metadata[image_id]['path'])
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} not found")
            
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Default to image/png for simplicity
        base64_data = base64.b64encode(image_data).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"
    
    def _save_metadata(self):
        """Save image metadata to disk."""
        try:
            with open(self.base_dir / 'metadata.json', 'w') as f:
                import json
                json.dump(self.metadata, f)
                
        except Exception as e:
            logger.error(f"Error saving image metadata: {e}")
            raise
    
    def _load_images(self):
        """Load image metadata from disk."""
        try:
            metadata_path = self.base_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    import json
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} images")
            else:
                self.metadata = {}
                logger.info("No image metadata found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading image metadata: {e}")
            self.metadata = {}
    
    def list_images(self) -> List[Dict]:
        """List all images with their metadata."""
        return [
            {
                'id': image_id,
                'name': meta['name'],
                'added': meta['added'],
                'has_description': bool(meta.get('description'))
            }
            for image_id, meta in self.metadata.items()
        ]
    
    def delete_image(self, image_id: str) -> bool:
        """Delete an image and its associated description document."""
        if image_id not in self.metadata:
            return False
        
        try:
            # Delete image file
            image_path = Path(self.metadata[image_id]['path'])
            if image_path.exists():
                image_path.unlink()
            
            # Delete description document
            doc_name = f"image_{image_id}.txt"
            if self.document_manager and doc_name in self.document_manager.chunks:
                del self.document_manager.chunks[doc_name]
                del self.document_manager.embeddings[doc_name]
                del self.document_manager.metadata[doc_name]
                self.document_manager._save_to_disk()
            
            # Delete metadata
            del self.metadata[image_id]
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting image {image_id}: {e}")
            return False

    async def update_description(self, image_id: str, description: str) -> bool: # Changed to async def
        """Update the description for an image."""
        if image_id not in self.metadata:
            return False
        
        try:
            # Update metadata
            self.metadata[image_id]['description'] = description
            self._save_metadata()
            
            # Update document
            doc_name = f"image_{image_id}.txt"
            if self.document_manager:
                # Remove old document if it exists
                if doc_name in self.document_manager.chunks:
                    del self.document_manager.chunks[doc_name]
                if doc_name in self.document_manager.embeddings:
                    del self.document_manager.embeddings[doc_name]
                
                # Add new document
                await self.document_manager.add_document(doc_name, description) # Added await
                
                # Add image reference to document metadata
                if doc_name in self.document_manager.metadata:
                    self.document_manager.metadata[doc_name]['image_id'] = image_id
                    self.document_manager.metadata[doc_name]['image_name'] = self.metadata[image_id]['name']
                    self.document_manager._save_to_disk()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating image description {image_id}: {e}")
            return False

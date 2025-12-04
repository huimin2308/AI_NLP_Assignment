import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.firestore import SERVER_TIMESTAMP
from typing import List, Dict

class FirebaseChatManager:
    def __init__(self, cred_path="nlp-assignment-c87d8-firebase-adminsdk-fbsvc-2eba3fbc80.json"):
        """
        Initialize Firebase connection for chat history storage
        
        Args:
            cred_path: Path to Firebase service account JSON file
        """
        # Initialize Firebase app if not already initialized
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        
        self.db = firestore.client()
    
    def create_chat_session(self, user_id: str, chat_name: str, file_name: str, preprocess: str, temp_path, model) -> str:
        """
        Create a new chat session for a user
        
        Args:
            user_id: Unique identifier for the user
            chat_name: Name for the chat session
            
        Returns:
            The ID of the newly created chat session
        """
        chat_ref = self.db.collection('users').document(user_id).collection('chats').document()
        
        chat_data = {
            'chat_name': chat_name,
            'file_name': file_name,
            'activeChat': True,
            'preprocess': preprocess,
            'temp_path': temp_path,
            'model': model,
            'created_at': SERVER_TIMESTAMP,
            'updated_at': SERVER_TIMESTAMP,
            'message_count': 0
        }
        
        chat_ref.set(chat_data)
        return chat_ref.id
    
    def add_message(self, user_id: str, chat_id: str, role: str, content: str) -> None:
        """
        Add a message to a chat session
        
        Args:
            user_id: User ID
            chat_id: Chat session ID
            role: 'user' or 'bot'
            content: Message content
        """
        chat_ref = self.db.collection('users').document(user_id).collection('chats').document(chat_id)
        
        message_data = {
            'role': role,
            'content': content,
            'timestamp': SERVER_TIMESTAMP
        }
        
        # Add message to subcollection
        chat_ref.collection('messages').add(message_data)
        
        # Update counters and timestamp
        chat_ref.update({
            'updated_at': SERVER_TIMESTAMP,
            'message_count': firestore.Increment(1)
        })

    def get_chat_messages(self, user_id: str, chat_id: str, limit: int = 100) -> List[Dict]:
        """
        Retrieve messages from a chat session
        
        Args:
            user_id: User ID
            chat_id: Chat session ID
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries with only role and content fields,
            sorted by timestamp (oldest first)
        """
        messages_ref = (self.db.collection('users')
                        .document(user_id)
                        .collection('chats')
                        .document(chat_id)
                        .collection('messages')
                        .order_by('timestamp')
                        .limit(limit))
        
        messages = []
        for doc in messages_ref.stream():
            msg_data = doc.to_dict()
            messages.append({
                'role': msg_data.get('role'),
                'content': msg_data.get('content'),
                'timestamp': msg_data.get('timestamp')
            })
        
        return messages

    
    def get_user_chats(self, user_id: str) -> List[Dict]:
        """
        Get all chat sessions for a user with basic info
        
        Args:
            user_id: User ID
            
        Returns:
            List of chat sessions with metadata (newest first)
        """
        chats_ref = (self.db.collection('users')
                    .document(user_id)
                    .collection('chats')
                    .order_by('created_at', direction=firestore.Query.DESCENDING))
        
        return [{'id': doc.id, **doc.to_dict()} for doc in chats_ref.stream()]
    
    def deactivate_previous_chats(self, user_id:str):
        chat_ref = self.db.collection('users').document(user_id).collection('chats')
    
        # Get all chat documents
        chats = chat_ref.stream()
        
        for chat in chats:
            chat.reference.update({
                'activeChat': False
            })
    
    def delete_chat(self, user_id: str, chat_id: str) -> None:
        """
        Delete a chat session and all its messages
        
        Args:
            user_id: User ID
            chat_id: Chat session ID
        """
        # First delete all messages (Firestore doesn't automatically delete subcollections)
        messages_ref = (self.db.collection('users')
                       .document(user_id)
                       .collection('chats')
                       .document(chat_id)
                       .collection('messages'))
        
        # Batch delete messages
        batch = self.db.batch()
        for doc in messages_ref.list_documents():
            batch.delete(doc)
        batch.commit()
        
        # Then delete the chat document
        self.db.collection('users').document(user_id).collection('chats').document(chat_id).delete()
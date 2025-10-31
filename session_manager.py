from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
from pathlib import Path
import uuid

class ChatSession:
    """Represents a single chat session with history"""
    
    def __init__(self, session_id: str = None, user_id: str = "default"):
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.created_at = datetime.now()
        self.history: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the session history"""
        message = {
            "role": role,  # 'user' or 'assistant'
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.history.append(message)
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get chat history, optionally limited to last N messages"""
        if limit:
            return self.history[-limit:]
        return self.history
    
    def get_context_window(self, window_size: int = 5) -> str:
        """Get formatted conversation context for the last N exchanges"""
        recent_history = self.get_history(limit=window_size * 2)
        
        context_parts = []
        for msg in recent_history:
            role = msg["role"].upper()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n\n".join(context_parts)
    
    def clear_history(self):
        """Clear the chat history"""
        self.history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "history": self.history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create session from dictionary"""
        session = cls(session_id=data["session_id"], user_id=data["user_id"])
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.history = data["history"]
        return session


class SessionManager:
    """Manages multiple user sessions with persistence"""
    
    def __init__(self, sessions_dir: str = "sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.active_sessions: Dict[str, ChatSession] = {}
    
    def create_session(self, user_id: str = "default") -> ChatSession:
        """Create a new chat session"""
        session = ChatSession(user_id=user_id)
        self.active_sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get an existing session"""
        # Try to get from active sessions
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from disk
        session = self.load_session(session_id)
        if session:
            self.active_sessions[session_id] = session
        
        return session
    
    def get_or_create_session(self, session_id: str = None, user_id: str = "default") -> ChatSession:
        """Get existing session or create new one"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session(user_id)
    
    def save_session(self, session: ChatSession):
        """Save session to disk"""
        session_file = self.sessions_dir / f"{session.session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2)
    
    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load session from disk"""
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ChatSession.from_dict(data)
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None
    
    def list_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """List all sessions for a user"""
        sessions = []
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get("user_id") == user_id:
                    sessions.append({
                        "session_id": data["session_id"],
                        "created_at": data["created_at"],
                        "message_count": len(data["history"])
                    })
            except Exception as e:
                print(f"Error reading session file {session_file}: {e}")
        
        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Remove from disk
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
    
    def cleanup_old_sessions(self, days: int = 30):
        """Delete sessions older than specified days"""
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                created_at = datetime.fromisoformat(data["created_at"])
                if created_at < cutoff_date:
                    session_file.unlink()
                    print(f"Deleted old session: {data['session_id']}")
            except Exception as e:
                print(f"Error cleaning up session file {session_file}: {e}")

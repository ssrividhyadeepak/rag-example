import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Activity, Database, Brain, FileText } from 'lucide-react';
import MessageBubble from './MessageBubble';
import SystemStatus from './SystemStatus';
import { sendQuery, checkSystemHealth } from '../services/api';
import './ChatInterface.css';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState({});
  const [showStatus, setShowStatus] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Load initial welcome message
    const welcomeMessage = {
      id: 'welcome',
      type: 'assistant',
      content: `👋 Hi! I'm your ATM Support Assistant. I can help you with:

🔍 **Investigating ATM failures and errors**
📊 **Analyzing transaction patterns**
🛠️ **Troubleshooting specific issues**
📈 **Performance monitoring and insights**

Try asking me something like:
• "Why did ATM 1123 fail today at 10 AM?"
• "Show me all DDL_EXCEEDED errors from yesterday"
• "What does NETWORK_ERROR mean?"
• "Analyze withdrawal patterns for the last 24 hours"`,
      timestamp: new Date(),
      isWelcome: true
    };
    setMessages([welcomeMessage]);

    // Check system health
    checkSystemHealth()
      .then(status => setSystemStatus(status))
      .catch(err => console.error('Health check failed:', err));
  }, []);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await sendQuery(inputValue);

      const assistantMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.response,
        timestamp: new Date(),
        confidence: response.confidence,
        sources: response.sources,
        responseTime: response.response_time,
        intent: response.intent
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `❌ Sorry, I encountered an error: ${error.message}. Please make sure the API server is running.`,
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleExampleClick = (example) => {
    setInputValue(example);
  };

  const clearChat = () => {
    setMessages([]);
  };

  const exampleQueries = [
    "Why did ATM 1123 fail today at 10 AM?",
    "Show me DDL_EXCEEDED errors from yesterday",
    "What does NETWORK_ERROR mean?",
    "Analyze withdrawal patterns for the last 24 hours",
    "Show me all timeout issues from ATM 5567",
    "What's causing high failure rates today?"
  ];

  return (
    <div className="chat-interface">
      {/* Header */}
      <div className="chat-header">
        <div className="header-left">
          <div className="assistant-avatar">
            <Bot size={24} />
          </div>
          <div className="header-info">
            <h1>ATM Support Assistant</h1>
            <span className="status-indicator">
              <Activity size={12} />
              Online
            </span>
          </div>
        </div>
        <div className="header-actions">
          <button
            className="header-btn"
            onClick={() => setShowStatus(!showStatus)}
            title="System Status"
          >
            <Database size={18} />
          </button>
          <button
            className="header-btn"
            onClick={clearChat}
            title="Clear Chat"
          >
            Clear
          </button>
        </div>
      </div>

      {/* System Status Panel */}
      {showStatus && (
        <SystemStatus
          status={systemStatus}
          onClose={() => setShowStatus(false)}
        />
      )}

      {/* Messages Container */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <Bot size={48} />
            <h3>Ready to help!</h3>
            <p>Ask me about ATM issues, errors, or analysis</p>
          </div>
        ) : (
          messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))
        )}

        {isLoading && (
          <div className="typing-indicator">
            <div className="message-bubble assistant">
              <Bot size={20} />
              <div className="typing-animation">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Example Queries */}
      {messages.length <= 1 && (
        <div className="example-queries">
          <h4>Try these examples:</h4>
          <div className="example-buttons">
            {exampleQueries.map((query, index) => (
              <button
                key={index}
                className="example-btn"
                onClick={() => handleExampleClick(query)}
              >
                {query}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Ask about ATM issues, errors, or analysis... (e.g., 'Why did ATM 1123 fail today at 10 AM?')"
            rows={1}
            disabled={isLoading}
          />
          <button
            className="send-button"
            onClick={handleSend}
            disabled={!inputValue.trim() || isLoading}
          >
            <Send size={20} />
          </button>
        </div>
        <div className="input-suggestions">
          <button onClick={() => setInputValue(inputValue + 'ATM ')}>ATM ID</button>
          <button onClick={() => setInputValue(inputValue + 'error code ')}>Error Code</button>
          <button onClick={() => setInputValue(inputValue + 'today at ')}>Today</button>
          <button onClick={() => setInputValue(inputValue + 'yesterday ')}>Yesterday</button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
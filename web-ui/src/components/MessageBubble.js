import React from 'react';
import { Bot, User, Clock, Target, FileText } from 'lucide-react';
import './MessageBubble.css';

const MessageBubble = ({ message }) => {
  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatContent = (content) => {
    // Convert markdown-like formatting to HTML
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/^• (.*$)/gim, '<li>$1</li>')
      .replace(/\n/g, '<br>');
  };

  const isUser = message.type === 'user';

  return (
    <div className={`message-container ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-bubble">
        <div className="message-header">
          <div className="message-avatar">
            {isUser ? <User size={18} /> : <Bot size={18} />}
          </div>
          <div className="message-meta">
            <span className="message-sender">
              {isUser ? 'You' : 'Assistant'}
            </span>
            <span className="message-time">
              <Clock size={12} />
              {formatTime(message.timestamp)}
            </span>
          </div>
        </div>

        <div className="message-content">
          {message.isWelcome ? (
            <div
              className="welcome-content"
              dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
            />
          ) : (
            <div
              className={`content-text ${message.isError ? 'error' : ''}`}
              dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
            />
          )}
        </div>

        {/* Assistant message metadata */}
        {!isUser && !message.isWelcome && !message.isError && (
          <div className="message-footer">
            {message.confidence && (
              <div className="confidence-indicator">
                <Target size={12} />
                <span>Confidence: {Math.round(message.confidence * 100)}%</span>
              </div>
            )}
            {message.responseTime && (
              <div className="response-time">
                <Clock size={12} />
                <span>{message.responseTime.toFixed(2)}s</span>
              </div>
            )}
            {message.sources && message.sources.length > 0 && (
              <div className="sources-indicator">
                <FileText size={12} />
                <span>{message.sources.length} source{message.sources.length !== 1 ? 's' : ''}</span>
              </div>
            )}
            {message.intent && (
              <div className="intent-indicator">
                <span className={`intent-badge ${message.intent}`}>
                  {message.intent.replace('_', ' ')}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;
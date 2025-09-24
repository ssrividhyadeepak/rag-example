import React from 'react';
import { X, CheckCircle, XCircle, AlertCircle, Activity } from 'lucide-react';
import './SystemStatus.css';

const SystemStatus = ({ status, onClose }) => {
  const getStatusIcon = (componentStatus) => {
    switch (componentStatus) {
      case 'healthy':
        return <CheckCircle size={16} className="status-healthy" />;
      case 'degraded':
        return <AlertCircle size={16} className="status-degraded" />;
      case 'unhealthy':
        return <XCircle size={16} className="status-unhealthy" />;
      default:
        return <Activity size={16} className="status-unknown" />;
    }
  };

  const formatUptime = (seconds) => {
    if (!seconds) return 'Unknown';

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  return (
    <div className="system-status-overlay">
      <div className="system-status-panel">
        <div className="status-header">
          <h3>System Status</h3>
          <button className="close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div className="status-content">
          {/* Overall Status */}
          <div className="overall-status">
            <div className="status-indicator">
              {getStatusIcon(status.status)}
              <span className={`status-text ${status.status}`}>
                {status.status || 'Unknown'}
              </span>
            </div>
            <div className="status-timestamp">
              Last checked: {status.timestamp ? new Date(status.timestamp).toLocaleTimeString() : 'Unknown'}
            </div>
          </div>

          {/* Component Status */}
          <div className="components-status">
            <h4>Components</h4>
            <div className="component-list">
              {status.components ? Object.entries(status.components).map(([component, info]) => (
                <div key={component} className="component-item">
                  <div className="component-info">
                    {getStatusIcon(info.status)}
                    <span className="component-name">
                      {component.charAt(0).toUpperCase() + component.slice(1)}
                    </span>
                  </div>
                  <div className="component-details">
                    {info.details}
                  </div>
                </div>
              )) : (
                <div className="no-components">
                  No component information available
                </div>
              )}
            </div>
          </div>

          {/* System Metrics */}
          <div className="system-metrics">
            <h4>Metrics</h4>
            <div className="metrics-grid">
              <div className="metric-item">
                <span className="metric-label">Uptime:</span>
                <span className="metric-value">
                  {formatUptime(status.uptime_seconds)}
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Version:</span>
                <span className="metric-value">
                  {status.version || 'Unknown'}
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Memory Usage:</span>
                <span className="metric-value">
                  {status.memory_usage || 'N/A'}
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Active Connections:</span>
                <span className="metric-value">
                  {status.active_connections || 'N/A'}
                </span>
              </div>
            </div>
          </div>

          {/* API Endpoints Status */}
          <div className="endpoints-status">
            <h4>API Endpoints</h4>
            <div className="endpoint-list">
              <div className="endpoint-item">
                <span className="endpoint-path">/api/v1/query</span>
                <span className="endpoint-status healthy">✓ Active</span>
              </div>
              <div className="endpoint-item">
                <span className="endpoint-path">/api/v1/search</span>
                <span className="endpoint-status healthy">✓ Active</span>
              </div>
              <div className="endpoint-item">
                <span className="endpoint-path">/api/v1/troubleshoot</span>
                <span className="endpoint-status healthy">✓ Active</span>
              </div>
              <div className="endpoint-item">
                <span className="endpoint-path">/api/v1/health</span>
                <span className="endpoint-status healthy">✓ Active</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemStatus;
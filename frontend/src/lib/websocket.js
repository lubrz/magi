/**
 * WebSocket client for TRIAD deliberation streaming
 */

export class TriadWS {
  constructor(url, onMessage, onError) {
    this.url = url;
    this.onMessage = onMessage;
    this.onError = onError;
    this.socket = null;
    this.reconnectAttempts = 0;
  }

  connect() {
    console.log(`Connecting to TRIAD WebSocket: ${this.url}`);
    
    try {
      this.socket = new WebSocket(this.url);
      
      this.socket.onopen = () => {
        console.log('TRIAD WebSocket connected');
        this.reconnectAttempts = 0;
        if (this.onMessage) this.onMessage({ type: 'system_status', data: { message: 'CONNECTED TO TRIAD CORE' } });
      };

      this.socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (this.onMessage) this.onMessage(message);
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket Error:', error);
        if (this.onError) this.onError(error);
      };

      this.socket.onclose = () => {
        console.log('WebSocket connection closed');
        this.attemptReconnect();
      };
    } catch (err) {
      console.error('Failed to initiate WebSocket:', err);
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < 5) {
      this.reconnectAttempts++;
      console.log(`Attempting reconnect ${this.reconnectAttempts}...`);
      setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
    }
  }

  send(data) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(data));
    } else {
      console.warn('Socket not open. Message queued or lost.');
    }
  }

  ask(question, maxRounds = 3) {
    this.send({ question, max_rounds: maxRounds });
  }
}

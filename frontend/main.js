// ============================================================================
// Offline Chatbot Frontend - Vanilla JavaScript
// ============================================================================

const API_URL = 'http://localhost:8000';

// State
let state = {
    sessionId: localStorage.getItem('chatbot_session_id') || generateSessionId(),
    isLoading: false,
    conversations: [],
};

// DOM Elements
const sidebar = document.getElementById('sidebar');
const conversationsList = document.getElementById('conversationsList');
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const btnSend = document.getElementById('btnSend');
const btnNewChat = document.getElementById('btnNewChat');
const btnNewChatSidebar = document.getElementById('btnNewChatSidebar');
const btnToggleSidebar = document.getElementById('btnToggleSidebar');
const sessionIdEl = document.getElementById('sessionId');
const statusText = document.getElementById('statusText');
const messageForm = document.getElementById('messageForm');

// ============================================================================
// INITIALIZATION
// ============================================================================

function init() {
    // Save session to localStorage
    localStorage.setItem('chatbot_session_id', state.sessionId);
    updateSessionDisplay();
    enableInput();
    checkHealth();
    loadConversations();
    
    // Event listeners
    btnNewChat.addEventListener('click', startNewChat);
    btnNewChatSidebar.addEventListener('click', startNewChat);
    messageForm.addEventListener('submit', handleSendMessage);
    messageInput.addEventListener('keydown', handleKeyDown);
    btnToggleSidebar.addEventListener('click', toggleSidebar);
}

// ============================================================================
// SIDEBAR MANAGEMENT
// ============================================================================

function toggleSidebar() {
    sidebar.classList.toggle('open');
}

function closeSidebar() {
    sidebar.classList.remove('open');
}

// ============================================================================
// CONVERSATION MANAGEMENT
// ============================================================================

async function loadConversations() {
    try {
        const response = await fetch(`${API_URL}/conversations`);
        if (!response.ok) {
            throw new Error('Falha ao carregar conversas');
        }
        
        state.conversations = await response.json();
        renderConversationsList();
    } catch (error) {
        console.error('Erro ao carregar conversas:', error);
    }
}

function renderConversationsList() {
    if (state.conversations.length === 0) {
        conversationsList.innerHTML = '<div class="placeholder">Nenhum chat ainda</div>';
        return;
    }

    conversationsList.innerHTML = state.conversations
        .map(conv => {
            const date = new Date(conv.updated_at * 1000);
            const timeStr = date.toLocaleDateString('pt-BR', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
            });

            const isActive = conv.id === state.sessionId ? 'active' : '';

            return `
                <div class="conversation-item ${isActive}" data-id="${conv.id}">
                    <div class="conversation-item-text" onclick="loadConversation('${conv.id}')">
                        <div class="conversation-item-title">${escapeHtml(conv.title)}</div>
                        <div class="conversation-item-date">${timeStr}</div>
                    </div>
                    <button 
                        class="conversation-item-delete" 
                        onclick="deleteConversation('${conv.id}', event)" 
                        title="Deletar"
                    >
                        🗑️
                    </button>
                </div>
            `;
        })
        .join('');
}

async function loadConversation(sessionId) {
    try {
        const response = await fetch(`${API_URL}/conversations/${sessionId}`);
        if (!response.ok) {
            throw new Error('Conversa não encontrada');
        }

        const conversation = await response.json();
        
        // Atualizar estado
        state.sessionId = sessionId;
        localStorage.setItem('chatbot_session_id', state.sessionId);
        updateSessionDisplay();

        // Atualizar UI
        chatMessages.innerHTML = '';
        
        if (conversation.messages.length === 0) {
            chatMessages.innerHTML = '<div class="message-system">Nenhuma mensagem nesta conversa</div>';
        } else {
            conversation.messages.forEach(msg => {
                addMessage(msg.content, msg.role === 'assistant' ? 'bot' : 'user', msg.timestamp);
            });
        }

        // Atualizar lista de conversas
        renderConversationsList();
        
        // Fechar sidebar em mobile
        closeSidebar();
        
        // Reabilitar input
        enableInput();
    } catch (error) {
        console.error('Erro ao carregar conversa:', error);
        setStatus('Erro ao carregar conversa', false);
    }
}

async function deleteConversation(sessionId, event) {
    event.stopPropagation();
    
    if (!confirm('Tem certeza que deseja deletar esta conversa?')) {
        return;
    }

    try {
        const response = await fetch(`${API_URL}/chat/${sessionId}`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
        });

        if (!response.ok) {
            throw new Error('Erro ao deletar conversa');
        }

        // Se deletou a conversa atual, criar nova
        if (sessionId === state.sessionId) {
            startNewChat();
        } else {
            // Senão, apenas recarregar lista
            await loadConversations();
        }
    } catch (error) {
        console.error('Erro ao deletar conversa:', error);
        setStatus('Erro ao deletar conversa', false);
    }
}

// ============================================================================
// SESSION MANAGEMENT
// ============================================================================

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function updateSessionDisplay() {
    const shortId = state.sessionId.substring(0, 16) + '...';
    sessionIdEl.textContent = `Sessão: ${shortId}`;
    sessionIdEl.title = state.sessionId;
}

async function startNewChat() {
    // Delete current session if it has no messages
    if (state.sessionId && chatMessages.children.length <= 1) {
        try {
            await fetch(`${API_URL}/chat/${state.sessionId}`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
            });
        } catch (error) {
            console.warn('Erro ao deletar sessão anterior:', error);
        }
    }

    // Create new session
    state.sessionId = generateSessionId();
    localStorage.setItem('chatbot_session_id', state.sessionId);
    updateSessionDisplay();

    // Clear chat
    chatMessages.innerHTML = '<div class="message-system">Nova conversa iniciada</div>';
    messageInput.value = '';
    messageInput.focus();

    // Recarregar conversas
    await loadConversations();
    
    // Fechar sidebar
    closeSidebar();
}

// ============================================================================
// API COMMUNICATION
// ============================================================================

async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            setStatus('Pronto', true);
            enableInput();
        } else {
            setStatus('Erro na conexão', false);
            disableInput();
        }
    } catch (error) {
        console.error('Health check failed:', error);
        setStatus('Desconectado', false);
        disableInput();
    }
}

async function sendMessage(message) {
    if (!message.trim() || state.isLoading) {
        return;
    }

    // Add user message to UI
    addMessage(message, 'user');
    messageInput.value = '';
    messageInput.focus();

    // Set loading state
    state.isLoading = true;
    disableInput();
    setStatus('Processando...', false);
    
    // Add loading indicator
    const loadingId = addLoadingMessage();

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                message: message,
            }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        
        // Remove loading indicator
        removeLoadingMessage(loadingId);
        
        // Add bot response
        addMessage(data.response, 'bot');
        
        // Recarregar conversas para atualizar sidebar
        await loadConversations();
        
        setStatus('Pronto', true);
    } catch (error) {
        console.error('Send message error:', error);
        removeLoadingMessage(loadingId);
        addMessage(
            'Desculpe, ocorreu um erro ao processar sua mensagem. Verifique se a API está rodando em http://localhost:8000',
            'bot'
        );
        setStatus('Erro', false);
    } finally {
        state.isLoading = false;
        enableInput();
    }
}

// ============================================================================
// UI HELPERS
// ============================================================================

function addMessage(text, sender, timestamp = null) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${sender}`;
    
    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';
    contentEl.textContent = text;
    
    messageEl.appendChild(contentEl);
    
    // Add timestamp
    const timeEl = document.createElement('span');
    timeEl.className = 'message-timestamp';
    
    if (timestamp) {
        const date = new Date(timestamp * 1000);
        timeEl.textContent = date.toLocaleTimeString('pt-BR', {
            hour: '2-digit',
            minute: '2-digit',
        });
    } else {
        timeEl.textContent = new Date().toLocaleTimeString('pt-BR', {
            hour: '2-digit',
            minute: '2-digit',
        });
    }
    
    contentEl.appendChild(timeEl);
    
    chatMessages.appendChild(messageEl);
    scrollToBottom();
}

function addLoadingMessage() {
    const id = 'loading_' + Date.now();
    const messageEl = document.createElement('div');
    messageEl.id = id;
    messageEl.className = 'message bot';
    
    const contentEl = document.createElement('div');
    contentEl.className = 'message-content loading';
    contentEl.innerHTML = `
        <span>Pensando</span>
        <span class="loading-dots">
            <span></span><span></span><span></span>
        </span>
    `;
    
    messageEl.appendChild(contentEl);
    chatMessages.appendChild(messageEl);
    scrollToBottom();
    
    return id;
}

function removeLoadingMessage(id) {
    const el = document.getElementById(id);
    if (el) {
        el.remove();
    }
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setStatus(text, isGood) {
    statusText.textContent = text;
    const indicator = document.querySelector('.status-indicator');
    if (isGood) {
        indicator.style.background = '#10b981';
    } else {
        indicator.style.background = '#ef4444';
    }
}

function enableInput() {
    messageInput.disabled = false;
    btnSend.disabled = false;
}

function disableInput() {
    messageInput.disabled = true;
    messageInput.blur();
    btnSend.disabled = true;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

function handleSendMessage(event) {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (message) {
        sendMessage(message);
    }
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleSendMessage(event);
    }
}

// ============================================================================
// STARTUP
// ============================================================================

document.addEventListener('DOMContentLoaded', init);

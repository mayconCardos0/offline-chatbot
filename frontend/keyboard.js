/* Teclado virtual para o modo kiosk (tela touch sem teclado físico).
   Ativado apenas quando a URL contém ?kiosk=1. */
(function () {
    'use strict';

    if (!new URLSearchParams(window.location.search).has('kiosk')) return;

    document.documentElement.classList.add('kiosk');
    document.addEventListener('contextmenu', (e) => e.preventDefault());

    const TEXT_INPUT = 'input:not([type]), input[type="text"], input[type="search"], textarea';
    const SHIFT_LABELS = { off: '⇧', once: '⬆', lock: '⇪' };

    const LAYERS = {
        letters: [
            'qwertyuiop'.split(''),
            'asdfghjklç'.split(''),
            [{ id: 'shift' }, ...'zxcvbnm'.split(''), { id: 'backspace' }],
            [{ id: 'symbols' }, ',', { id: 'space' }, '.', { id: 'enter' }],
        ],
        symbols: [
            '1234567890'.split(''),
            ['-', '/', ':', ';', '(', ')', '$', '&', '@', '"'],
            [{ id: 'accents' }, '.', ',', '?', '!', "'", '%', { id: 'backspace' }],
            [{ id: 'letters' }, { id: 'space' }, { id: 'enter' }],
        ],
        accents: [
            'áéíóúãõâêô'.split(''),
            'ÁÉÍÓÚÃÕÂÊÔ'.split(''),
            [{ id: 'symbols' }, ...'àÀçÇü'.split(''), { id: 'backspace' }],
            [{ id: 'letters' }, { id: 'space' }, { id: 'enter' }],
        ],
    };

    const SPECIAL = {
        shift: { cls: 'fn' },
        backspace: { cls: 'fn', label: '⌫' },
        symbols: { cls: 'fn', label: '?123' },
        accents: { cls: 'fn', label: 'áé' },
        letters: { cls: 'fn', label: 'ABC' },
        space: { cls: 'space', label: 'espaço' },
        enter: { cls: 'enter' },
    };

    let layer = 'letters';
    let shift = 'off';
    let target = null;
    let holdTimer = null;
    let repeatInterval = null;

    const root = document.createElement('div');
    root.className = 'vkbd';
    root.hidden = true;
    root.innerHTML =
        '<div class="vkbd-bar"><button type="button" class="vkbd-close" tabindex="-1">⌄ Fechar</button></div>' +
        '<div class="vkbd-keys"></div>';
    document.body.appendChild(root);
    const keysEl = root.querySelector('.vkbd-keys');

    function labelFor(id) {
        if (id === 'shift') return SHIFT_LABELS[shift];
        if (id === 'enter') return target && target.id === 'messageInput' ? 'Enviar' : 'OK';
        return SPECIAL[id].label;
    }

    function render() {
        keysEl.textContent = '';
        LAYERS[layer].forEach((row) => {
            const rowEl = document.createElement('div');
            rowEl.className = 'vkbd-row';
            row.forEach((key) => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.tabIndex = -1;
                btn.className = 'vkbd-key';
                if (typeof key === 'string') {
                    const ch = layer === 'letters' && shift !== 'off' ? key.toUpperCase() : key;
                    btn.textContent = ch;
                    btn.dataset.char = ch;
                } else {
                    btn.classList.add(SPECIAL[key.id].cls);
                    if (key.id === 'shift' && shift !== 'off') btn.classList.add('on');
                    btn.textContent = labelFor(key.id);
                    btn.dataset.action = key.id;
                }
                rowEl.appendChild(btn);
            });
            keysEl.appendChild(rowEl);
        });
    }

    function fireInput(el) {
        el.dispatchEvent(new Event('input', { bubbles: true }));
    }

    function insertText(text) {
        const el = target;
        if (!el) return;
        const start = el.selectionStart ?? el.value.length;
        const end = el.selectionEnd ?? start;
        el.setRangeText(text, start, end, 'end');
        fireInput(el);
    }

    function deleteBack() {
        const el = target;
        if (!el) return;
        const start = el.selectionStart ?? el.value.length;
        const end = el.selectionEnd ?? start;
        if (start !== end) {
            el.setRangeText('', start, end, 'end');
        } else if (start > 0) {
            el.setRangeText('', start - 1, start, 'end');
        } else {
            return;
        }
        fireInput(el);
    }

    function pressEnter() {
        const el = target;
        if (!el) return;
        const ev = new KeyboardEvent('keydown', {
            key: 'Enter',
            code: 'Enter',
            bubbles: true,
            cancelable: true,
        });
        const notHandled = el.dispatchEvent(ev);
        if (notHandled && el.form) el.form.requestSubmit();
        setTimeout(() => {
            if (!el.isConnected || root.hidden) return;
            if (layer === 'letters' && shift === 'off' && !el.value.trim()) {
                shift = 'once';
                render();
            }
        }, 80);
    }

    function press(btn) {
        if (btn.dataset.char !== undefined) {
            insertText(btn.dataset.char);
            if (shift === 'once') {
                shift = 'off';
                render();
            }
            return;
        }
        switch (btn.dataset.action) {
            case 'shift':
                shift = shift === 'off' ? 'once' : shift === 'once' ? 'lock' : 'off';
                render();
                break;
            case 'backspace':
                deleteBack();
                break;
            case 'space':
                insertText(' ');
                break;
            case 'enter':
                pressEnter();
                break;
            default:
                layer = btn.dataset.action;
                render();
        }
    }

    function stopRepeat() {
        clearTimeout(holdTimer);
        clearInterval(repeatInterval);
        holdTimer = null;
        repeatInterval = null;
    }

    function show(el) {
        const wasHidden = root.hidden;
        target = el;
        if (wasHidden) {
            layer = 'letters';
            shift = 'off';
        }
        if (layer === 'letters' && shift === 'off' && !el.value.trim()) shift = 'once';
        render();
        root.hidden = false;
        document.documentElement.style.setProperty('--kbd-height', root.offsetHeight + 'px');
        document.body.classList.add('kbd-open');
        requestAnimationFrame(() => el.scrollIntoView({ block: 'nearest' }));
    }

    function hide() {
        stopRepeat();
        target = null;
        root.hidden = true;
        document.body.classList.remove('kbd-open');
    }

    // Mantém o foco no campo de texto: sem isso o blur dispara antes da tecla.
    root.addEventListener('mousedown', (e) => e.preventDefault());
    root.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        const btn = e.target.closest('button');
        if (!btn) return;
        if (btn.classList.contains('vkbd-close')) {
            if (target) target.blur();
            return;
        }
        press(btn);
        if (btn.dataset.action === 'backspace') {
            stopRepeat();
            holdTimer = setTimeout(() => {
                repeatInterval = setInterval(deleteBack, 70);
            }, 450);
        }
    });
    ['pointerup', 'pointercancel', 'pointerleave'].forEach((type) => {
        root.addEventListener(type, stopRepeat);
    });

    document.addEventListener('focusin', (e) => {
        const el = e.target;
        if (el.matches && el.matches(TEXT_INPUT) && !el.disabled && !el.readOnly) show(el);
    });

    document.addEventListener('focusout', () => {
        setTimeout(() => {
            const active = document.activeElement;
            if (!(active && active.matches && active.matches(TEXT_INPUT))) hide();
        }, 0);
    });

    // O campo de renomear é removido do DOM ao confirmar, sem disparar focusout.
    setInterval(() => {
        if (!root.hidden && (!target || !target.isConnected || target.disabled)) hide();
    }, 500);
})();

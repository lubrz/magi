/**
 * CRT & Terminal Animations
 */

export const typeWriter = (element, text, speed = 20) => {
  let i = 0;
  element.innerHTML = '';
  
  return new Promise((resolve) => {
    function type() {
      if (i < text.length) {
        // Handle line breaks
        if (text.charAt(i) === '\n') {
          element.innerHTML += '<br/>';
        } else {
          element.innerHTML += text.charAt(i);
        }
        i++;
        setTimeout(type, speed);
      } else {
        resolve();
      }
    }
    type();
  });
};

export const glitchEffect = (element) => {
  element.classList.add('glitch');
  setTimeout(() => {
    element.classList.remove('glitch');
  }, 500);
};

export const shakeElement = (element) => {
  element.classList.add('shake');
  setTimeout(() => {
    element.classList.remove('shake');
  }, 400);
};

/**
 * Update the status indicator pulse speed
 */
export const setPulseSpeed = (speed = '2s') => {
  const indicator = document.getElementById('indicator-pulse');
  if (indicator) {
    indicator.style.animationDuration = speed;
  }
};

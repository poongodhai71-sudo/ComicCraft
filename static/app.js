const description = document.querySelector('#description');
const characterCount = document.querySelector('#character-count');
const sampleButton = document.querySelector('#sample-button');
const storyForm = document.querySelector('#story-form');

if (description && characterCount) {
  const updateCount = () => {
    characterCount.textContent = `${description.value.length} / 600`;
  };
  description.addEventListener('input', updateCount);
}

if (sampleButton) {
  sampleButton.addEventListener('click', () => {
    const sample = {
      title: 'The Lantern at Low Tide',
      character: 'Mira',
      description: 'Mira finds a tiny lantern beneath the old pier. A shy sea dragon asks for help finding its way home. Together they follow a trail of blue lights across the bay.',
      character_details: 'A curious young lighthouse keeper in a golden raincoat.',
      setting: 'A quiet harbor beneath a violet twilight sky',
      genre: 'Fantasy',
      tone: 'Inspiring',
      art_style: 'Anime',
      panel_count: '4',
    };
    for (const [name, value] of Object.entries(sample)) {
      const field = storyForm?.querySelector(`[name="${name}"]`);
      if (field) field.value = value;
    }
    if (description && characterCount) {
      characterCount.textContent = `${description.value.length} / 600`;
    }
    document.querySelector('[name="title"]')?.focus();
  });
}

const generationStatus = document.querySelector('#ai-generation-status');
const generationMessage = document.querySelector('#ai-generation-message');
const generationProgress = document.querySelector('#ai-generation-progress');

function showGenerationStatus(message, progress = 0, isError = false) {
  if (!generationStatus || !generationMessage || !generationProgress) return;
  generationStatus.hidden = false;
  generationStatus.classList.toggle('is-error', isError);
  generationMessage.textContent = message;
  generationProgress.value = progress;
  generationProgress.hidden = isError;
  generationStatus.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

for (const form of document.querySelectorAll('[data-ai-generation]')) {
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;

    const submitButton = form.querySelector('button[type="submit"]');
    if (submitButton) submitButton.disabled = true;
    const formData = new FormData(form);
    if (event.submitter?.name) formData.set(event.submitter.name, event.submitter.value);
    const demoMode = formData.get('generation_mode') === 'demo';
    showGenerationStatus(demoMode ? 'Preparing your clearly labeled demo story…' : 'Connecting to Gemini…', 2);
    let completed = false;

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        body: formData,
        headers: { Accept: 'text/event-stream' },
        credentials: 'same-origin',
      });
      if (!response.ok) {
        const result = await response.json().catch(() => ({}));
        throw new Error(result.message || 'Gemini could not start this generation. No comic was created.');
      }
      if (!response.body) throw new Error('This browser cannot stream generation progress. Please use a current browser.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        let boundary = buffer.indexOf('\n\n');
        while (boundary !== -1) {
          const eventText = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const dataLine = eventText.split('\n').find((line) => line.startsWith('data:'));
          if (dataLine) {
            const payload = JSON.parse(dataLine.slice(5).trim());
            if (payload.type === 'progress') {
              showGenerationStatus(payload.message, payload.progress || 0);
            } else if (payload.type === 'error') {
              throw new Error(payload.message || 'Gemini could not finish this comic. No incomplete comic was saved.');
            } else if (payload.type === 'complete' && payload.generation_id) {
              completed = true;
              window.location.assign(`/preview?generation_id=${encodeURIComponent(payload.generation_id)}`);
              return;
            }
          }
          boundary = buffer.indexOf('\n\n');
        }
        if (done) break;
      }
      throw new Error('Gemini closed the generation stream before the comic was complete. No comic was saved.');
    } catch (error) {
      showGenerationStatus(error instanceof Error ? error.message : 'Comic generation failed. Please try again.', 0, true);
    } finally {
      if (submitButton && !completed) submitButton.disabled = false;
    }
  });
}

for (const button of document.querySelectorAll('.panel-regenerate')) {
  button.addEventListener('click', async () => {
    const panel = button.closest('.comic-panel');
    const image = panel?.querySelector('.panel-image');
    const status = panel?.querySelector('.panel-regeneration-status');
    if (!panel || !image || !status) return;

    button.disabled = true;
    status.textContent = 'Requesting a replacement image from Gemini…';
    try {
      const response = await fetch(button.dataset.url, { method: 'POST', credentials: 'same-origin' });
      const result = await response.json().catch(() => ({}));
      if (!response.ok || !result.image_url) {
        throw new Error(result.message || 'Gemini did not return a replacement image. The existing panel is unchanged.');
      }
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = () => reject(new Error('The regenerated image could not be loaded. The previous image is unchanged.'));
        image.src = result.image_url;
      });
      status.textContent = 'New AI-generated image ready.';
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : 'Image regeneration failed. The existing panel is unchanged.';
    } finally {
      button.disabled = false;
    }
  });
}

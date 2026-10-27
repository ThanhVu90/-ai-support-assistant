const $ = id => document.getElementById(id);
let draftText = '';
function element(tag, text, className) { const node = document.createElement(tag); node.textContent = text; if(className) node.className = className; return node; }
async function api(path, data) {
  const response = await fetch(path, data ? {method:'POST',headers:{'Content-Type':'application/json','X-Support-Client':'web'},body:JSON.stringify(data)} : {});
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Please check the text and try again.');
  return body;
}
async function initialize() {
  try {
    const [health, documents] = await Promise.all([api('/api/health'), api('/api/documents')]);
    $('mode').textContent = health.mode === 'ai' ? 'AI evidence selection' : 'Document search · No AI calls';
    $('mode-help').textContent = health.mode === 'ai' ? 'Question and retrieved excerpts are sent to OpenAI.' : 'Local search. No AI calls or external data transfer.';
    $('count').textContent = `${documents.length} documents`;
    for (const doc of documents) {
      const card = element('article','','doc');
      card.append(element('div','▤','icon'),element('strong',doc.title),element('small',`${doc.chunks} excerpts · ${doc.revision.slice(0,8)}`));
      $('documents').append(card);
    }
  } catch { $('mode').textContent = 'Service unavailable'; }
}
$('ask-form').addEventListener('submit', async event => {
  event.preventDefault(); $('search').disabled = true;
  $('result').replaceChildren(element('p','Searching the knowledge base…','result-head'));
  try {
    const data = await api('/api/ask',{question:$('question').value.trim()});
    const label = data.status === 'search_results' ? 'Matching excerpts — relevance requires your review' : data.status === 'evidence_selected' ? 'AI-selected evidence — verify before acting' : 'No supporting evidence';
    $('result').replaceChildren(element('div',`${label} · ${data.elapsed_ms} ms`,'result-head'));
    $('result').append(element('p',data.message));
    for (const item of data.evidence) {
      const box = element('article','','evidence');
      box.append(element('h3',item.title),element('blockquote',item.quote),element('small',`${item.source} / ${item.id} / revision ${item.revision}`));
      $('result').append(box);
    }
  } catch (error) { $('result').replaceChildren(element('div',error.message,'error')); }
  finally { $('search').disabled=false; }
});
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click',()=>{ $('question').value=button.dataset.question; $('ask-form').requestSubmit(); }));
$('draft-form').addEventListener('submit',async event=>{
  event.preventDefault();
  if (!$('question').reportValidity()) return;
  try {
    const draft = await api('/api/drafts',{question:$('question').value.trim(),impact:$('impact').value.trim()});
    draftText = `DRAFT — NOT SUBMITTED\n${draft.title}\n\n${draft.description}`;
    $('draft').textContent=draftText; $('download').hidden=false;
  } catch(error) { $('draft').textContent=error.message; $('download').hidden=true; }
});
$('download').addEventListener('click',()=>{
  const url=URL.createObjectURL(new Blob([draftText],{type:'text/plain;charset=utf-8'}));
  const anchor=document.createElement('a');anchor.href=url;anchor.download='support-ticket-draft.txt';anchor.click();URL.revokeObjectURL(url);
});
initialize();

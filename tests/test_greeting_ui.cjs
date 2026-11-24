// Run the actual greeting lifecycle functions with controlled network timing.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('static/index.html','utf8');
new vm.Script(html.split('<script>')[1].split('</script>')[0]);
const code=html.slice(html.indexOf('let arrival='),html.indexOf('function textBubble('));
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {resolve,promise}};
function setup(){
 const calls=[],shown=[],listeners={},windowListeners={},timers=[],store=new Map(),text={value:''};
 const node=()=>({setAttribute(){},remove(){this.removed=true}});
 const list={scrollTop:0,scrollHeight:2000,clientHeight:400,append(...nodes){shown.push(...nodes)},querySelector(){return null}};
 const s={messages:[{id:"prior",role:"user",text:"previous"}],consent:true,id:'u1',sessions:[{id:'s1',created:new Date().toISOString()}]};
 const ctx=vm.createContext({Promise,Date,String,Number,crypto:require('node:crypto').webcrypto,
  state:s,sid:'s1',token:'test',initialOpeningPending:false,returnOpeningPending:false,visitId:'v1',visitStarted:false,greetingEpoch:0,greetingPending:null,greetingCommit:Promise.resolve(),hiddenAt:0,busy:false,asset:null,
  $:id=>id==='text'?text:id==='messages'?list:{},document:{visibilityState:'visible',createElement:node,addEventListener:(n,f)=>listeners[n]=f},window:{addEventListener:(n,f)=>windowListeners[n]=f},setInterval(){},setTimeout:(f,ms)=>{let t={f,ms};timers.push(t);return t},clearTimeout:t=>{if(t)t.cancelled=true},hasUnsentInput:()=>!!(text.value||ctx.asset),
  localStorage:{getItem:k=>store.get(k),setItem:(k,v)=>store.set(k,v)},followBottom(){},timeSeparator:()=>node(),textBubble:(r,t)=>({...node(),text:t}),updateSurveySchedule(){},updateComposer(){},render(){calls.push('render')},localDay:x=>new Date(x).toDateString(),fail:e=>{throw e},
  api:async(path,body)=>{calls.push(path);return path==='/api/state'?s:{created:true}},
 });vm.runInContext(code,ctx);return {ctx,calls,shown,listeners,windowListeners,timers,store,text,list};
}
(async()=>{
 // User starts typing while a greeting response is in flight: no display/commit.
 {const t=setup(),wait=deferred();t.ctx.api=async p=>{t.calls.push(p);return p==='/api/greeting'?wait.promise:{ok:true}};
  const opening=t.ctx.startVisit();t.ctx.cancelGreeting();wait.resolve({prepared:true,draft:'d1',reply:'晚到的问候'});await opening;
  assert(!t.calls.includes('/api/greeting/commit'));assert(!t.shown.some(n=>n.text));assert(t.shown.every(n=>n.removed));}
 // Display happens before commit starts, and only once. No full-history refresh.
 {const t=setup();t.ctx.api=async p=>{t.calls.push(p);if(p==='/api/greeting')return {prepared:true,draft:'d1',reply:'回来啦～\n歇一会儿吧'};assert(t.shown.some(n=>n.text==='回来啦～'));return {created:true}};
  await t.ctx.startVisit();await t.ctx.startVisit();assert.equal(t.calls.filter(p=>p==='/api/greeting').length,1);assert(!t.calls.includes('/api/state'));}
 // Refreshing history after a chat never asks for an opening.
 {const t=setup();await t.ctx.refresh();assert(!t.calls.includes('/api/greeting'));}
 // A recent open is silent; returning after ten minutes permits an opening.
 {const t=setup();t.store.set('xiaoyuLastSeen:u1',String(Date.now()-590000));await t.ctx.openVisit();assert(!t.calls.includes('/api/greeting'));
  t.store.set('xiaoyuLastSeen:u1',String(Date.now()-610000));t.ctx.captureArrival();await t.ctx.openVisit();assert(t.calls.includes('/api/greeting'));}
 // Hiding the page invalidates an in-flight draft even if it completes later.
 {const t=setup(),wait=deferred();t.ctx.api=async p=>{t.calls.push(p);return p==='/api/greeting'?wait.promise:{ok:true}};
  const opening=t.ctx.startVisit();t.ctx.document.visibilityState='hidden';t.listeners.visibilitychange();wait.resolve({prepared:true,draft:'d1',reply:'未显示'});await opening;assert(!t.calls.includes('/api/greeting/commit'));}
 // A foreground refresh must not wipe a user message sent during the state fetch.
 {const t=setup(),wait=deferred();t.ctx.api=()=>wait.promise;const loading=t.ctx.refresh(true);await Promise.resolve();t.ctx.cancelGreeting();t.ctx.busy=true;wait.resolve(t.ctx.state);await loading;assert(!t.calls.includes('render'));}
 // Slow session/history loading must not let a presence heartbeat erase the absence.
 {const t=setup();t.store.set('xiaoyuLastSeen:u1',String(Date.now()-610000));
  t.ctx.rememberPresence();await t.ctx.openVisit();assert(t.calls.includes('/api/greeting'),'heartbeat erased return eligibility');}
 // A state response started before a greeting appeared must not erase that greeting.
 {const t=setup(),wait=deferred();t.ctx.api=async p=>{t.calls.push(p);if(p==='/api/state')return wait.promise;
  if(p==='/api/greeting')return {prepared:true,draft:'d1',reply:'回来啦～'};return {created:true}};
  const refreshing=t.ctx.refresh();await Promise.resolve();await Promise.resolve();await t.ctx.startVisit();
  wait.resolve(t.ctx.state);await refreshing;assert(!t.calls.includes('render'),'stale history overwrote visible greeting');}
 // Entering a chat with older scroll position still presents the opening at the bottom.
 {const t=setup();t.ctx.api=async p=>p==='/api/greeting'?{prepared:true,draft:'d1',reply:'回来啦～'}:{created:true};
  await t.ctx.startVisit();assert.equal(t.list.scrollTop,t.list.scrollHeight);}
 // A second opening request in the same foreground visit must not repeat a greeting.
 {const t=setup();t.store.set('xiaoyuLastSeen:u1',String(Date.now()-610000));
  await t.ctx.openVisit();await t.ctx.openVisit();assert.equal(t.calls.filter(p=>p==='/api/greeting').length,1);}
 // History refresh must not remove an opening's immediate typing indicator.
 {const t=setup(),wait=deferred();t.ctx.api=async p=>{t.calls.push(p);return p==='/api/greeting'?wait.promise:p==='/api/state'?t.ctx.state:{created:true}};
  const opening=t.ctx.startVisit();await t.ctx.refresh();assert(!t.calls.includes('render'));
  t.ctx.cancelGreeting();wait.resolve({prepared:true,draft:'d1',reply:'晚到的问候'});await opening;assert(!t.calls.includes('/api/greeting/commit'));}
 // Mobile return while an old reply is busy keeps its arrival until the reply settles.
 {const t=setup();t.ctx.document.visibilityState='hidden';t.listeners.visibilitychange();t.ctx.hiddenAt=Date.now()-610000;t.ctx.busy=true;t.ctx.document.visibilityState='visible';
  await t.ctx.resumeOpening(true);assert(!t.calls.includes('/api/greeting'));t.ctx.busy=false;await t.ctx.resumeOpening();assert.equal(t.calls.filter(p=>p==='/api/greeting').length,1);}
 // User typing after returning cancels the pending opening; a later refresh cannot revive it.
 {const t=setup();t.ctx.hiddenAt=Date.now()-610000;t.ctx.busy=true;await t.ctx.resumeOpening(true);t.ctx.cancelGreeting();t.ctx.busy=false;await t.ctx.resumeOpening();await t.ctx.refresh();assert(!t.calls.includes('/api/greeting'));}
 // A frozen mobile browser without hide/show events is recovered by a heartbeat gap.
 {const t=setup();await t.ctx.openVisit();t.calls.length=0;vm.runInContext('lastHeartbeatAt=Date.now()-610000',t.ctx);await t.ctx.resumeOpening();assert.equal(t.calls.filter(p=>p==='/api/greeting').length,1);await t.ctx.resumeOpening();assert.equal(t.calls.filter(p=>p==='/api/greeting').length,1);}
 // Both pageshow and visibility can fire for the same return: one request only.
 {const t=setup(),wait=deferred();t.ctx.hiddenAt=Date.now()-610000;t.ctx.api=async p=>{t.calls.push(p);return p==='/api/state'?wait.promise:{created:true}};
  const a=t.ctx.resumeOpening(true),b=t.ctx.resumeOpening(true);await Promise.resolve();wait.resolve(t.ctx.state);await Promise.all([a,b]);assert.equal(t.calls.filter(p=>p==='/api/greeting').length,1);}
 // A foreground history fetch can retry after a brief network failure.
 {const t=setup();t.ctx.hiddenAt=Date.now()-610000;let failed=false;t.ctx.api=async p=>{t.calls.push(p);if(p==='/api/state'&&!failed){failed=true;throw Error('offline')}return p==='/api/state'?t.ctx.state:{created:true}};
  await t.ctx.resumeOpening(true);assert.equal(t.timers.filter(t=>!t.cancelled).length,1);await t.ctx.resumeOpening();assert(t.calls.includes('/api/greeting'));}
 // A failed generation may retry, but never after the user begins a new turn.
 {const t=setup();t.ctx.api=async p=>{t.calls.push(p);throw Error('offline')};await t.ctx.startVisit(610);assert.equal(t.timers.filter(t=>!t.cancelled).length,1);t.ctx.cancelGreeting();assert(t.timers.every(t=>t.cancelled));await t.timers.at(-1).f();assert.equal(t.calls.filter(p=>p==='/api/greeting').length,1);}
 // A new account opens immediately despite a recent presence heartbeat and
 // never enters the return-model path; the composer is held until ready.
 {const t=setup();t.ctx.state.messages=[];t.store.set('xiaoyuLastSeen:u1',String(Date.now()));let complete=false;
  t.ctx.api=async p=>{t.calls.push(p);if(p==='/api/opening'){assert(t.ctx.initialOpeningPending);complete=true;return {created:true}}
   return complete?{...t.ctx.state,messages:[{id:'welcome',role:'assistant',text:'嗨，我是小屿～'}]}:t.ctx.state};
  await t.ctx.refresh(true);assert(t.calls.includes('/api/opening'));assert(!t.calls.includes('/api/greeting'));assert(!t.ctx.initialOpeningPending);assert.equal(t.ctx.state.messages[0].id,'welcome');}
 // Post-send history refresh cannot create the first opening even with stale empty history.
 {const t=setup();t.ctx.state.messages=[];await t.ctx.refresh(false);assert(!t.calls.includes('/api/opening'));}
 // No first opening is created before consent.
 {const t=setup();t.ctx.state.messages=[];t.ctx.state.consent=false;t.ctx.$=id=>({showModal(){},hidden:false});
  await t.ctx.refresh(true);assert(!t.calls.includes('/api/opening'));}
 // A new-account opening failure must not leave controls held or masquerade as a greeting.
 {const t=setup();t.ctx.state.messages=[];t.ctx.api=async p=>{t.calls.push(p);if(p==='/api/opening')throw Error('offline');return t.ctx.state};
  await assert.rejects(t.ctx.refresh(true),/offline/);assert(!t.ctx.initialOpeningPending);assert(!t.calls.includes('/api/greeting'));}
 // Slow generation holds controls; deadline fallback appears first and late API is discarded.
 {const t=setup(),wait=deferred();t.ctx.api=async(p,b)=>{t.calls.push(p);
  if(p==='/api/greeting')return wait.promise;
  if(p==='/api/greeting/fallback'){assert.equal(b.baseline_message_id,'prior');return {prepared:true,draft:'fast',reply:'回来啦～'}}
  return {created:true}};
  const opening=t.ctx.startVisit(610);assert(t.ctx.returnOpeningPending);
  t.timers.find(x=>x.ms===8000).f();await opening;
  assert(!t.ctx.returnOpeningPending);assert.equal(t.shown.filter(x=>x.text&&!x.removed).length,1);
  wait.resolve({prepared:true,draft:'late',reply:'迟到'});await Promise.resolve();await Promise.resolve();
  assert(!t.shown.some(x=>x.text==='迟到'));assert.equal(t.calls.filter(x=>x==='/api/greeting/commit').length,1);}
 // Total fallback network failure releases controls; a departure never starts fallback.
 {const t=setup();t.ctx.api=async()=>{throw Error('offline')};await t.ctx.startVisit(610);assert(!t.ctx.returnOpeningPending);}
 {const t=setup(),wait=deferred();t.ctx.api=async p=>{t.calls.push(p);return p==='/api/greeting'?wait.promise:{ok:true}};
  const opening=t.ctx.startVisit(610);t.ctx.cancelGreeting();await opening;assert(!t.ctx.returnOpeningPending);
  t.timers.find(x=>x.ms===8000).f();await Promise.resolve();await Promise.resolve();assert(!t.calls.includes('/api/greeting/fallback'));}
 // Initial history loading also holds controls, so typing cannot race the opening.
 {const t=setup(),wait=deferred();t.ctx.api=async p=>p==='/api/state'?wait.promise:{created:false};
  const opening=t.ctx.refresh(true);assert(t.ctx.returnOpeningPending);wait.resolve(t.ctx.state);await opening;assert(!t.ctx.returnOpeningPending);}
 console.log('PASS: 25 greeting UI race/lifecycle checks + full inline JS syntax');
})().catch(e=>{console.error(e);process.exitCode=1});

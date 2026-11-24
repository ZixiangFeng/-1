const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('static/index.html','utf8');new vm.Script(html.split('<script>')[1].split('</script>')[0]);
const code=html.slice(html.indexOf('function hasUnsentInput()'),html.indexOf('const COMMON_EMOJIS'));
function setup(){
 let now=100000,next=0,timers=new Map(),nodes={};
 for(const id of ['text','preview','libraryDialog','file','status'])nodes[id]={value:'',childElementCount:0,open:false,addEventListener(){},click(){}};
 const c=vm.createContext({Promise,Date:class extends Date{static now(){return now}},Error,
  $:id=>nodes[id],initialOpeningPending:false,returnOpeningPending:false,asset:null,uploadPending:false,filePicking:false,imeComposing:false,busy:true,activeTurn:'t1',activeTyping:null,pauseUntil:0,pauseTimer:null,interruptionRequested:false,
  interruptedTurns:new Set(),document:{querySelectorAll:()=>[]},window:{addEventListener(){}},cancelGreeting(){},rememberPresence(){},updateComposer(){},api:async()=>({ok:true}),
  setTimeout:(fn,delay)=>{timers.set(++next,{fn,at:now+delay});return next},clearTimeout:id=>timers.delete(id)});vm.runInContext(code,c);
 async function advance(ms){now+=ms;for(const [id,t] of [...timers])if(t.at<=now){timers.delete(id);t.fn()}await Promise.resolve();await Promise.resolve()}
 return {c,nodes,advance};
}
(async()=>{
 for(const kind of ['text','asset','upload','picker','dialog','ime','preview']){
  const t=setup();if(kind==='text')t.nodes.text.value='还没发完';if(kind==='asset')t.c.asset='img';if(kind==='upload')t.c.uploadPending=true;if(kind==='picker')t.c.filePicking=true;if(kind==='dialog')t.nodes.libraryDialog.open=true;if(kind==='ime')t.c.imeComposing=true;if(kind==='preview')t.nodes.preview.childElementCount=1;
  t.c.pauseForInput();let done=false;const result=t.c.waitForReplyPermission('t1').then(v=>{done=true;return v});await t.advance(10000);assert.equal(done,false,kind+' resumed too early');
  t.nodes.text.value='';t.c.asset=null;t.c.uploadPending=false;t.c.filePicking=false;t.nodes.libraryDialog.open=false;t.c.imeComposing=false;t.nodes.preview.childElementCount=0;
  t.c.pauseForInput();await t.advance(1900);assert.equal(done,false);await t.advance(200);assert.equal(await result,true);
 }
 const t=setup();t.nodes.text.value='追加';t.c.pauseForInput();const waiting=t.c.waitForReplyPermission('t1');t.c.interruptedTurns.add('t1');await t.advance(100);assert.equal(await waiting,false);
 const both=setup();both.nodes.text.value='文字';both.c.asset='sticker';both.c.pauseForInput();both.nodes.text.value='';both.c.pauseForInput();await both.advance(10000);assert(both.c.hasUnsentInput());assert(both.c.interruptionRequested);
 console.log('PASS: 7 draft types held beyond timeout; cancel grace; superseded turn; text+image partial cancellation');
})().catch(e=>{console.error(e);process.exitCode=1});

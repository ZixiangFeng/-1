const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('static/index.html','utf8');new vm.Script(html.split('<script>')[1].split('</script>')[0]);
const code=html.slice(html.indexOf('// Received-sticker actions:'),html.indexOf('function imageBubble'));
let now=0,next=0,timers=new Map(),opens=[],requests=[],resolveSave;
function element(){return {listeners:{},classList:{add(){}},style:{},isConnected:true,open:false,hidden:true,addEventListener(n,f){this.listeners[n]=f},setAttribute(){},getBoundingClientRect(){return {left:25,bottom:60,width:220,height:110}},showModal(){this.open=true},close(){this.open=false;this.listeners.close?.({})}}}
let nodes=Object.fromEntries(['stickerActionDialog','saveChatSticker','cancelStickerAction','chatToast','libraryDialog'].map(k=>[k,element()]));
const c=vm.createContext({Math,token:'test-auth',$:id=>nodes[id],window:{innerWidth:400,innerHeight:700},matchMedia:()=>({matches:false}),loadLibrary:async()=>{},api:async(path,body)=>{requests.push({path,body});return new Promise(r=>resolveSave=r)},setTimeout:(fn,delay)=>{timers.set(++next,{fn,at:now+delay});return next},clearTimeout:id=>timers.delete(id)});
vm.runInContext(code,c);c.openStickerAction=(message,p)=>opens.push({message,p});
function advance(ms){now+=ms;for(const [id,t] of [...timers])if(t.at<=now){timers.delete(id);t.fn()}}
function event(extra={}){return {pointerType:'touch',pointerId:1,isPrimary:true,clientX:30,clientY:40,preventDefault(){this.prevented=true},stopPropagation(){},...extra}}
function frame(){let x=element();x.dispose=()=>{x.disposed=true};c.bindStickerActions(x,'received-message');return x}
(async()=>{
 let f=frame();f.listeners.pointerdown(event());advance(549);assert.equal(opens.length,0);advance(1);assert.equal(opens.length,1);assert.equal(opens[0].message,'received-message');
 f.listeners.pointerdown(event());f.listeners.pointermove(event({clientY:65}));advance(600);assert.equal(opens.length,1,'scroll must not open');
 f.listeners.pointerdown(event());f.listeners.pointerup(event());advance(600);assert.equal(opens.length,1,'tap must not open');
 f.listeners.pointerdown(event());f.listeners.pointercancel(event());advance(600);assert.equal(opens.length,1);
 f.listeners.pointerdown(event());f.dispose();advance(600);assert.equal(opens.length,1);assert(f.disposed);
 f=frame();let right=event();f.listeners.contextmenu(right);assert(right.prevented);assert.equal(opens.length,2);
 f.listeners.keydown(event({key:'Enter'}));assert.equal(opens.length,3);
 f.listeners.pointerdown(event({pointerType:'mouse'}));advance(600);assert.equal(opens.length,3);
 vm.runInContext("stickerAction={message:'received-message',auth:token}",c);
 const saved=nodes.saveChatSticker.onclick();await nodes.saveChatSticker.onclick();assert.equal(requests.length,1);assert.equal(requests[0].path,'/api/library/save');assert.equal(requests[0].body.message,'received-message');
 resolveSave({duplicate:false});await saved;assert.equal(nodes.chatToast.textContent,'已添加到我的表情');assert.equal(nodes.stickerActionDialog.open,false);
 vm.runInContext("stickerAction={message:'received-message',auth:token}",c);
 const again=nodes.saveChatSticker.onclick();resolveSave({duplicate:true});await again;assert.equal(nodes.chatToast.textContent,'已在我的表情中');
 console.log('PASS: long press, right click, keyboard, scroll/tap/cancel, disposal, duplicate requests and save feedback');
})().catch(e=>{console.error(e);process.exitCode=1});

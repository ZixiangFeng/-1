const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let html=fs.readFileSync('static/research.html','utf8').replace('__SCREENSHOT_IMPORT__',fs.readFileSync('static/research-import.html','utf8')).replace('__GENERIC_SCREENSHOT_IMPORT__',fs.readFileSync('static/research-generic-import.html','utf8'));
new vm.Script([...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).join('\n'));
const code=fs.readFileSync('static/research-generic-import.html','utf8').split('<script>')[1].split('</script>')[0];
const nodes=new Map(),node=()=>({disabled:false,hidden:false,dataset:{},textContent:'',addEventListener(){},replaceChildren(){}});
const get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id)};
const ctx=vm.createContext({document:{getElementById:get,querySelectorAll:()=>[]},genericRefresh:async()=>{}});
vm.runInContext(code,ctx);
const state=vm.runInContext('genericCrops',ctx);
function setup(){state.pages=[{draft:'frozen-preview',name:'sheet',tiles:[0,1,2].map(index=>({index,row:1,column:index+1,selected:true,status:'pending',box:[0,0,20,20],card:node(),statusNode:node(),inputs:[node()]}))}];state.busy=false;state.stop=false;}
(async()=>{
 setup();let calls=[],fail=true;
 ctx.api=async(path,b)=>{calls.push(b);if(b.index===1&&fail)throw Error('temporary');return{id:'g_'+b.index,duplicate:false}};
 await ctx.importGenericCrops();assert.deepEqual(calls.map(x=>x.index),[0,1,2]);
 assert.equal(state.pages[0].tiles[0].status,'done');assert.equal(state.pages[0].tiles[1].status,'failed');assert.equal(state.pages[0].tiles[2].status,'done');
 fail=false;await ctx.importGenericCrops();assert.deepEqual(calls.map(x=>x.index),[0,1,2,1]);assert(calls.every(x=>x.draft==='frozen-preview'));
 assert.equal(get('genericCropImport').disabled,true);assert.equal(get('genericShotFiles').disabled,false);
 setup();calls=[];let resolve;
 ctx.api=async(path,b)=>{calls.push(b.index);return new Promise(r=>resolve=r)};
 const work=ctx.importGenericCrops();assert.equal(get('genericShotFiles').disabled,true);
 ctx.pauseGenericCrops();resolve({id:'g_0',duplicate:false});await work;assert.deepEqual(calls,[0]);
 ctx.api=async(path,b)=>{calls.push(b.index);return{id:'g_'+b.index,duplicate:true}};
 await ctx.importGenericCrops();assert.deepEqual(calls,[0,1,2]);assert.equal(get('genericShotFiles').disabled,false);
 ctx.clearGenericCrops();assert.equal(state.pages.length,0);assert.equal(get('genericCropImport').textContent,'一键上传并分析');
 console.log('PASS: combined admin script syntax, partial failure/retry, pause/resume, frozen draft, control restoration');
})().catch(e=>{console.error(e);process.exitCode=1});

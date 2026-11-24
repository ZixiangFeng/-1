"""Isolated headless Chrome checks; never attaches to the user's Chrome session."""
import base64, functools, http.server, json, subprocess, sys, tempfile, threading, time
from pathlib import Path
import requests, websocket
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sticker-broad-v5-20261119'


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass


def main():
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(OUT/'web')))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:%s/index.html'%server.server_port
    suffix='public' if len(sys.argv)>1 else 'local'
    with tempfile.TemporaryDirectory(prefix='xiaoyu-v5-qa-') as profile:
        process=subprocess.Popen(['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','--headless','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--remote-allow-origins=*','--host-resolver-rules=MAP www.hustdesign.top 39.107.159.82','--user-data-dir='+profile,'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            f=Path(profile)/'DevToolsActivePort'
            for _ in range(100):
                if f.exists():break
                time.sleep(.1)
            port=f.read_text().splitlines()[0]
            tab=next(t for t in requests.get('http://127.0.0.1:'+port+'/json/list',timeout=10).json() if t['type']=='page')
            ws=websocket.create_connection(tab['webSocketDebuggerUrl'],timeout=30);sequence=0;exceptions=[]
            def call(method,params=None):
                nonlocal sequence
                sequence+=1;n=sequence;ws.send(json.dumps({'id':n,'method':method,'params':params or {}}))
                while True:
                    message=json.loads(ws.recv())
                    if message.get('method')=='Runtime.exceptionThrown':exceptions.append(message)
                    if message.get('id')==n:
                        if 'error' in message:raise RuntimeError(message['error'])
                        return message.get('result',{})
            def js(expression,awaited=False):
                result=call('Runtime.evaluate',{'expression':expression,'returnByValue':True,'awaitPromise':awaited})
                if 'exceptionDetails' in result:raise RuntimeError(result)
                return result.get('result',{}).get('value')
            call('Page.enable');call('Runtime.enable');call('Page.navigate',{'url':url})
            ready="new Promise((resolve,reject)=>{let n=0;const timer=setInterval(()=>{if(window.v5Result){clearInterval(timer);resolve(true)}else if(n++>200){clearInterval(timer);reject('not ready')}},100)})"
            js(ready,True)
            info={'url':url,'examples':[]}
            for width,height in [(1365,960),(390,844)]:
                call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':width<500})
                js("document.querySelector('#queryCard').scrollIntoView({block:'start'})")
                js("Promise.all([...document.querySelectorAll('img')].filter(x=>x.getBoundingClientRect().top<innerHeight&&x.getBoundingClientRect().bottom>0).map(x=>x.decode().catch(()=>null)))",True)
                state=js("(()=>{const r=window.v5Result;return {width:innerWidth,scrollWidth:document.documentElement.scrollWidth,query:r.query,stage1:r.stage1.length,stage2:r.stage2.length,ids:r.stage2.map(x=>x.id),groups:r.query_groups,failedVisible:[...document.images].filter(x=>x.complete&&!x.naturalWidth).map(x=>x.alt)}})()")
                assert state['scrollWidth']<=width+1,state
                assert state['query']=='B0032' and state['stage1']>0 and state['stage2']>0,state
                assert not set(state['ids'])&{'B0128','B0314','B0403'},state
                assert not state['failedVisible'],state
                info['examples'].append(state)
                (OUT/(suffix+'-'+str(width)+'.png')).write_bytes(base64.b64decode(call('Page.captureScreenshot',{'format':'png'})['data']))
            js("document.getElementById('intent').value='ack';document.getElementById('intent').dispatchEvent(new Event('change'))")
            js(ready,True)
            assert js("window.v5Result.query_groups[0]")=='ack'
            js("document.getElementById('search').value='B0032';document.getElementById('search').dispatchEvent(new Event('input'))")
            js(ready,True)
            assert js("document.getElementById('query').options.length")==1
            assert not exceptions,exceptions
            info['intent_switch']=True;info['search']=True;info['js_errors']=0
            (OUT/(suffix+'-qa.json')).write_text(json.dumps(info,ensure_ascii=False,indent=2));print(json.dumps(info,ensure_ascii=False))
            ws.close()
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
            server.shutdown()


if __name__=='__main__':main()

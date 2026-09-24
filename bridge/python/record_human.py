"""Dedicated native-game recorder and localhost side panel. No training starts here."""
import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time

from isaac_bridge.human_recording import archive_episode, make_replay

REPO = Path(__file__).resolve().parents[2]
ALLOWED = {'prepare', 'summon', 'start', 'reset', 'stop', 'finish_session'}


def read_status(root):
    path = root/'status.jsonl'
    if not path.exists():
        return dict(mode='offline', error='Waiting for recorder Mod', command_id=0)
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-16384))
        lines = stream.read().split(b'\n')
    # Only parse a complete trailing line; concurrent append may leave a partial one.
    result = json.loads(lines[-2])
    if time.time()-result['utc'] > 5:
        result.update(mode='offline', error='Recorder heartbeat is stale; do not start playing')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--port', type=int, default=8764)
    parser.add_argument('--qa', action='store_true')
    parser.add_argument('--no-launch', action='store_true')
    args = parser.parse_args()
    root = (args.out or REPO/'runs/human'/datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'replays').mkdir()
    state = dict(root=str(root),episodes=[],pending=False)
    lock = threading.Lock(); issued = 0; validated = set()
    errors = root/'service-errors.log'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, payload, kind='application/json; charset=utf-8'):
            data = json.dumps(payload,ensure_ascii=False).encode('utf8') if isinstance(payload,dict) else payload
            self.send_response(code);self.send_header('Content-Type',kind)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':
                self.reply(200,(Path(__file__).parent/'isaac_bridge/human_panel.html').read_bytes(),'text/html; charset=utf-8')
            elif self.path == '/api/status':
                with lock:
                    native=read_status(root)
                    self.reply(200,{**state,**native,'pending':issued>native.get('command_id',0)})
            elif re.fullmatch(r'/replay/episode-\d{4}\.html',self.path):
                p=root/'replays'/self.path.rsplit('/',1)[1]
                self.reply(200,p.read_bytes(),'text/html; charset=utf-8') if p.exists() else self.reply(404,{'error':'Replay not ready'})
            else:
                self.reply(404,{'error':'Not found'})

        def do_POST(self):
            nonlocal issued
            if self.path!='/api/command' or self.headers.get('Content-Type')!='application/json':
                self.reply(400,{'error':'Expected JSON command'});return
            if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{args.port}',f'http://localhost:{args.port}'):
                self.reply(403,{'error':'Cross-origin commands are forbidden'});return
            try:
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                name=payload['command']
                permitted=ALLOWED | ({'qa_script','qa_win','qa_timeout'} if args.qa else set())
                if name not in permitted:raise ValueError('Unknown command')
                with lock:
                    native=read_status(root)
                    if native['mode'] in ('offline','closed'):raise ValueError('Recorder is not connected or session is closed')
                    if name in ('start','summon','qa_script','qa_timeout') and native['mode']!='ready':
                        raise ValueError('Prepare/reset the room first')
                    if issued>native['command_id']:raise ValueError('Previous command not yet acknowledged')
                    issued+=1
                    p=root/f'command-{issued:06d}.json';temp=p.with_suffix('.tmp')
                    temp.write_text(json.dumps({'command':name}));os.replace(temp,p)
                self.reply(200,{'id':issued})
            except (ValueError,KeyError) as exc:
                self.reply(409,{'error':str(exc)})

    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    if not args.no_launch:
        from isaac_bridge.parallel import prepare_worker
        from isaac_bridge.launch import DEFAULT_GAME_DIR
        from isaac_bridge.turbo import launch_suspended
        runtime,profile,savedata=prepare_worker(root/'worker',DEFAULT_GAME_DIR,Path.home()/'Documents/My Games/Binding of Isaac Repentance+')
        (runtime/'mods/isaac_rl_bridge/disable.it').touch()
        shutil.copytree(REPO/'bridge/mod/isaac_human_recorder',runtime/'mods/isaac_human_recorder')
        options=(savedata/'options.ini').read_text(encoding='utf8')
        for key,value in {'WindowWidth':1100,'WindowHeight':750,'WindowPosX':20,'WindowPosY':60,'Fullscreen':0,'EnableDebugConsole':1}.items():
            options=re.sub(rf'(?m)^{key}=.*$',f'{key}={value}',options)
        (savedata/'options.ini').write_text(options,encoding='utf8')
        os.environ['ISAAC_HUMAN_DIR']=str(root).replace('\\','/')
        os.environ['ISAAC_HUMAN_QA']='1' if args.qa else '0'
        pid,control,output=launch_suspended(27991,game_dir=str(runtime),privileged=False,
            extra_args=('--luadebug','--set-stage=1'),skip_render=False,virtual_clock=False,
            font_guard=False,file_retry=False,probe_dump=False,worker_profile=str(profile),log_dir=str(root/'native'))
        control.close()
        (root/'launch.json').write_text(json.dumps(dict(pid=pid,root=str(root),qa=args.qa,output=output,port=args.port),indent=2),encoding='utf8')
    print(json.dumps({'url':f'http://127.0.0.1:{args.port}','root':str(root)},ensure_ascii=False),flush=True)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    while True:
        # Validate only closed episodes. An unfinished/crashed capture stays raw and untrusted.
        for path in root.glob('episode-*.jsonl'):
            if path.name in validated:continue
            with path.open('rb') as stream:
                stream.seek(max(0,path.stat().st_size-512));tail=stream.read().split(b'\n')
            if len(tail)<2 or b'"end"' not in tail[-2]:continue
            try:
                result=archive_episode(path,encode_actor=True)
                make_replay(path,root/'replays'/path.with_suffix('.html').name)
            except (ValueError,OSError,KeyError) as exc:
                result={'file':path.name,'valid':False,'error':str(exc)}
                with errors.open('a',encoding='utf8') as stream:stream.write(repr(exc)+'\n')
            with lock:
                state['episodes'].append(result);validated.add(path.name)
                (root/'manifest.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
        time.sleep(.5)


if __name__=='__main__':main()

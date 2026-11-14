import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const python=path.join(root,'backend/.venv/bin/python');
if(!existsSync(python)){console.error('Run npm run setup:backend first.');process.exit(1);}
let stopping=false;const children=[];
function stop(code=0){if(stopping)return;stopping=true;for(const child of children)child.kill('SIGTERM');setTimeout(()=>process.exit(code),400);}
for(const [command,args] of [[python,['-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port','8000']], [process.execPath,['node_modules/vite/bin/vite.js','--host','127.0.0.1','--port','5174','--strictPort']]]){
 const child=spawn(command,args,{cwd:root,stdio:'inherit',env:{...process.env,PYTHONDONTWRITEBYTECODE:'1'}});children.push(child);child.on('error',err=>{console.error(err.message);stop(1);});child.on('exit',code=>{if(!stopping)stop(code||0);});
}
process.on('SIGINT',()=>stop());process.on('SIGTERM',()=>stop());

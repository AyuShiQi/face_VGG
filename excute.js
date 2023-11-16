const spawn = require('child_process').spawn
const py = spawn('python',['view_test.py'])

console.log('start…………')

py.stdout.on('data',function(res){
    let data = res.toString();
    console.log('stdout: ',data)
})
py.stderr.on('data',function(res){
    let data = res.toString();
    console.log('stderr: ',data)
})
py.on('close', (code) => {
  console.log(`子进程退出：退出代码code ${code}`);
});

console.log('end.')
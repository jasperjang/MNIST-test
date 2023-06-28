from git import Repo

# gets directory of given file path
def getDirPath(path):
    dirPath = ''
    pathList = path.split('/')
    for item in pathList:
        if item == '':
            pathList.remove(item)
    for i in range(len(pathList)-1):
        dirPath += f'/{pathList[i]}'
    return dirPath

path = '/home/jasperjang/MNIST-test/clearMLtest.py'
dirPath = getDirPath(path)
print(dirPath)
repo = Repo(f'{dirPath}')
print(repo)
current_branch = repo.head.reference.name
print(current_branch)
remote_url = repo.remotes.origin.url
print(remote_url)
project_name = remote_url.split("/")[-1].split(".")[0]
print(project_name)
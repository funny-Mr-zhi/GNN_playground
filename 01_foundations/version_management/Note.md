在线学习资料：
[gitbranch](https://learngitbranching.js.org/?locale=zh_CN)

* git commit
> 保存新的快照为一个完整的提交
* git branch
> 在现有提交上建立新的分支
* git checkout <branch name>
> 切换到分支
* git merge
> 合并分支，将所在分支头和指定分支头合并为一个新的提交
* git rebase
> 重定基，将现有分支头的历史提交链接在指定分支的后面，形成若干新的提交

HEDA
> 仅表示当前所在分支，对于branch -f命令，移动的是已有的分支名，HEAD通过checkout移动
* git log
> 查看所有提交的信息，包括哈希值。对于`main`这类提交，可以使用main作为相对引用
* git checkout HEAD^
> HEAD转移到当前提交的父节点，`HEAD^^`表示向上两个节点
* git checkout HEAD~4
> 回溯4次
* git branch -f <branch> <new-position>
> 移动分支
* git reset <tag>
> 分支回到指定的从前的提交
* git revert <tag>
> 创建新的提交，代码状态中，撤销指定tag相对与其上一个提交的更改

移动提交记录
* git cherry-pick <tag> <tag>
> 将指定tag提交按顺序放到当前branch下
* git rebase -i HEAD~x 
> 交互式移动当前分支到HEAD~x之后的区间的提交
* git commit --amend
> 对当前提交做出调整，创建当前提交的并行（同父级）提交
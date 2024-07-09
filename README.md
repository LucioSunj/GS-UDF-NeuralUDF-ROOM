# 配置完毕后，合作开发流程
本地首先创建了 `dev1` branch 用于本地开发
#Step 
1. 修改代码
2. Commit 到本地仓库
3. Checkout `master` branch
4. 将 `dev1` merge 到 `master`
5. pull 一下远程分支
6. 将本地分支 `master` push 到远程仓库
#Notice 
- log 中
	- 黄色为当前分支
	- 绿色为本地分支
	- 紫色为远程分支

# 代码冲突
- 显示冲突
#Step 
1. 点击 merge
2. 在中间窗口处结合两边代码不同之处修改
3. push
4. pull 更新

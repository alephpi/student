#!/usr/bin/zsh
# 提示用户输入 GitHub 用户名
echo "请输入你的 GitHub 用户名: "
read username

# 提示用户输入目标仓库名称
# read -p "请输入目标仓库名称: " target_repo

# 克隆目标仓库
# git clone git@github.com:$username/$target_repo.git
# cd $target_repo

# 函数：合并子仓库到目标仓库
merge_repo() {
    repo_name=$1
    echo "合并 $repo_name ..."
    # mkdir $repo_name

    # 添加远程仓库
    git remote add $repo_name git@github.com:$username/$repo_name.git

    # 获取远程仓库
    if ! git fetch $repo_name; then
        echo "错误：无法获取 $repo_name 的内容"
        git remote remove $repo_name
        return 1
    fi

    # 检查远程仓库的主分支名称
    if git branch -r | grep "$repo_name/main" > /dev/null; then
        branch="main"
    elif git branch -r | grep "$repo_name/master" > /dev/null; then
        branch="master"
    else
        echo "错误：无法确定 $repo_name 的主分支名称"
        git remote remove $repo_name
        return 1
    fi

    # 将远程仓库的内容合并到子目录
    if ! git subtree add --prefix=$repo_name $repo_name $branch; then
        echo "错误：无法将 $repo_name 合并到子目录"
        git remote remove $repo_name
        return 1
    fi

    # 移除远程仓库
    git remote remove $repo_name
}

# 提示用户输入要合并的仓库名，空行结束输入
echo "请输入要合并的仓库名称（每行一个，输入空行结束）："
repos=()
while read repo_name; do
    [[ -z "$repo_name" ]] && break
    repos+=("$repo_name")
done

# 合并每个仓库
for repo in "${repos[@]}"; do
    merge_repo $repo
done

# 推送合并后的仓库到 GitHub
git push origin main

echo "所有仓库已成功合并到 $target_repo."
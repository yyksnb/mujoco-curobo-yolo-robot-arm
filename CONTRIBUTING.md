# Contributing / 多人协作说明

这个仓库用于机械臂比赛/科研工程协作。为了减少冲突、保护可运行版本，所有
成员都按下面流程工作。

## 分支规则

- 不要直接向 `main` 分支提交代码。
- `main` 只保留稳定版本，应该随时可以被队友拉取和复现实验。
- 每个任务都从最新 `main` 新建自己的分支。
- 分支命名建议：
  - `feature/xxx`：新功能或新阶段工作
  - `fix/xxx`：bug 修复
  - `docs/xxx`：文档、报告、说明
  - `experiment/xxx`：实验性验证或临时方案

示例：

```bash
git checkout main
git pull
git checkout -b feature/real-robot-config
```

## 提交流程

1. 在自己的分支上修改。
2. 提交前运行相关脚本或测试。
3. 写清楚 commit message，简洁描述本次修改。
4. 推送分支到 GitHub。
5. 创建 Pull Request。
6. PR 合并前至少需要一名成员 review。
7. 测试通过后再合并。

示例 commit message：

```text
Add real robot config validator
Fix MuJoCo executor report path
Document cuRobo validation result
```

## Pull Request 要求

PR 里请说明：

- 改了什么。
- 为什么要改。
- 跑了哪些测试或脚本。
- 是否影响 Stage 1/2/3 pipeline。
- 是否修改文档。
- 是否包含大文件、数据、模型权重。

文档、实验结果、代码修改都应该走 PR。不要把本地临时结果直接推到 `main`。

## 测试要求

能运行全量测试时，优先运行：

```bash
pytest -q
```

如果只改了文档，也至少确认 Markdown 路径和命令没有明显错误。若测试失败，
不要把失败说成成功；在 PR 或总结里写清失败命令、关键报错和下一步建议。

## 大文件和数据

不要直接提交以下内容到 Git：

- 模型权重
- 数据集
- 大型 mesh 包
- 录屏、视频、大型图片
- 本地生成的 `outputs/`
- `__pycache__/`、`.pytest_cache/`

如果确实需要共享大文件，先和团队确认方案，例如 Git LFS、网盘、实验服务器
路径或单独的数据仓库。

## 冲突处理

如果 GitHub 提示冲突：

```bash
git checkout main
git pull
git checkout <your-branch>
git merge main
```

解决冲突后重新运行相关测试，再推送分支。不要为了省事直接覆盖队友的改动。

## GitHub Flow and main branch protection

Use GitHub Flow for team integration work:

1. Start from the latest stable `main`.
2. Create a scoped branch such as `feature/yolo-camera-capture`,
   `feature/object-assets`, `feature/robot-mujoco-scene`,
   `fix/stage3-adapter-validation`, or `docs/team-notes`.
3. Commit only source, tests, examples, configs, and docs that belong to the
   task.
4. Push the branch and open a Pull Request.
5. Run `pytest -q` or the task-specific validation commands before review.
6. Merge only after review and passing tests.

Keep `main` stable. Do not push directly to `main`, and do not commit
`outputs/`, `__pycache__/`, `.pytest_cache/`, model weights, datasets, temporary
logs, or large generated assets.

Raw YOLO/BODex formats should not replace the repository's Stage 3 canonical
contract. If an upstream format is different, add an adapter or converter and
document any mock pose or stub behavior clearly.

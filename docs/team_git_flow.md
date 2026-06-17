# Team GitHub Flow

## 原则

- `main` 是稳定分支，必须保持可运行。
- 不要直接 push `main`。
- 每个任务单独开分支，Pull Request 合并。
- 不提交 `outputs/`、`__pycache__/`、`.pytest_cache/`、模型权重、大数据文件。

## 标准流程

```bash
git clone https://github.com/yyksnb/mujoco-curobo-yolo-robot-arm.git
cd mujoco-curobo-yolo-robot-arm
git checkout main
git pull
git checkout -b feature/your-task
```

开发后：

```bash
pytest -q
git status
git add <changed-files>
git commit -m "short clear message"
git push -u origin feature/your-task
```

然后在 GitHub 创建 PR。

## 分支命名

- `feature/...`：新功能、新接入
- `fix/...`：修复 bug
- `docs/...`：文档、说明、流程
- `experiment/...`：实验性验证

示例：

```text
feature/yolo-camera-capture
feature/mujoco-object-assets
feature/robot-scene-config
feature/curobo-world-adapter
docs/team-integration-readability
```

## PR 检查

PR 描述里至少包含：

- 修改内容
- 修改原因
- 测试命令和结果
- 是否影响 Stage 1/2/3 pipeline
- 是否新增 raw format
- 是否新增 adapter/converter
- 是否包含大文件或运行产物

合并前：

- 至少 1 名队友 review。
- `pytest -q` 通过，或明确说明当前环境不能运行的原因。
- 不要把缓存或 `outputs/` 加进 commit。

## 冲突处理

```bash
git checkout main
git pull
git checkout <your-branch>
git merge main
```

解决冲突后重新运行相关测试，再 push 分支。

不要用强制覆盖的方式吞掉队友改动。


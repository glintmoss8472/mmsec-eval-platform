// 文件说明：该文件属于前端业务工具，集中实现 glossaryRegistry 相关逻辑。
import { attackCatalog } from "./attackCatalog";
import { datasetCatalog } from "./datasetCatalog";
import { recommendedModels } from "./modelCatalog";

export type GlossaryCategory =
  | "platform"
  | "page"
  | "section"
  | "workflow"
  | "attack"
  | "model"
  | "dataset"
  | "mode"
  | "parameter"
  | "metric"
  | "status"
  | "chart";

export interface GlossaryDetailSection {
  title: string;
  paragraphs: string[];
}

export interface GlossaryFormulaVariable {
  symbol: string;
  meaning: string;
}

export interface GlossaryFormulaBlock {
  title: string;
  latex: string;
  explanation: string[];
  variables?: GlossaryFormulaVariable[];
}

export interface GlossaryChartExplainBlock {
  title: string;
  items: Array<{ label: string; detail: string }>;
}

export interface GlossaryEntry {
  id: string;
  label: string;
  shortLabel: string;
  aliases: string[];
  category: GlossaryCategory;
  anchor: string;
  sourcePages: Array<"dashboard" | "testing" | "layout" | "glossary">;
  detailSections: GlossaryDetailSection[];
  formulaBlocks?: GlossaryFormulaBlock[];
  chartExplainBlocks?: GlossaryChartExplainBlock[];
  relatedIds?: string[];
}

/** 中文注释：实现 makeEntry 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
function makeEntry(
  prefix: string,
  slug: string,
  value: Omit<GlossaryEntry, "anchor">,
): GlossaryEntry {
  return {
    ...value,
    anchor: `entry-${prefix}-${slug}`,
  };
}

const staticEntries: GlossaryEntry[] = [
  makeEntry("platform", "main", {
    id: "platform-main",
    label: "面向多模态大模型的对抗安全测评平台（以图文检索为核心）",
    shortLabel: "平台定位",
    aliases: ["平台", "多模态大模型安全测评平台"],
    category: "platform",
    sourcePages: ["layout", "dashboard", "testing", "glossary"],
    detailSections: [
      {
        title: "这个名称在系统里指什么",
        paragraphs: [
          "它指的是当前这套前后端联动的多模态大模型对抗安全测评系统，但目前最稳定、最便于复查的核心部分，仍然是以图文检索任务为主的测评流程。",
          "页面顶部反复展示这个名称，是为了先让评委看清平台定位，也把范围说清楚：你做的是测评工具，而不是声称所有多模态任务都已经完成同等强度的验证。",
        ],
      },
      {
        title: "这个平台的具体功能是什么",
        paragraphs: [
          "平台负责统一发起对抗测试、调用模型、汇总攻击指标、生成运行结果，并把结果整理成答辩可直接展示的图表和术语说明。当前正式主证据和报告结构都围绕图文检索任务展开。",
        ],
      },
    ],
    relatedIds: ["page-dashboard", "page-testing", "page-glossary"],
  }),
  makeEntry("page", "dashboard", {
    id: "page-dashboard",
    label: "总览",
    shortLabel: "总览",
    aliases: [],
    category: "page",
    sourcePages: ["layout", "dashboard", "glossary"],
    detailSections: [
      {
        title: "这个页面负责什么",
        paragraphs: [
          "总览页只回答最核心的四件事：平台定位、攻击方法矩阵、受测模型矩阵、数据集矩阵，以及最近已经真实跑通的结果。",
          "它不是调参页，也不是调试页，所以保留的是老师能快速看懂的核心信息。",
        ],
      },
    ],
    relatedIds: ["section-attack-matrix", "section-model-matrix", "section-dataset-matrix", "section-latest-results"],
  }),
  makeEntry("page", "testing", {
    id: "page-testing",
    label: "对抗测试",
    shortLabel: "对抗测试",
    aliases: ["01 对抗测试"],
    category: "page",
    sourcePages: ["layout", "testing", "glossary"],
    detailSections: [
      {
        title: "这个页面负责什么",
        paragraphs: [
          "对抗测试是整个平台唯一的任务发起入口，用来选择攻击方法、代理模型、受测模型、数据集与参数，然后启动真实任务。",
          "页面右侧会同时显示当前任务进度、预计时间、最近验证结果和结果解读图表，因此它既负责发任务，也负责解释结果。",
        ],
      },
    ],
    relatedIds: ["workflow-job-progress", "mode-standard-eval"],
  }),
  makeEntry("page", "glossary", {
    id: "page-glossary",
    label: "术语与指标详解",
    shortLabel: "术语页",
    aliases: ["02 术语与指标详解", "说明页"],
    category: "page",
    sourcePages: ["layout", "glossary"],
    detailSections: [
      {
        title: "这个页面负责什么",
        paragraphs: [
          "它专门解释 `00 总览` 和 `01 对抗测试` 中出现的全部专有名词、参数、指标与图表。只要主页面里出现某个专业词，就应该能从这里看到对应的详细解释。",
          "你可以把它理解成答辩版的术语说明索引：支持搜索、跳转和高亮，避免老师问到术语时只能临场口头解释。",
        ],
      },
    ],
  }),
  makeEntry("section", "attack-matrix", {
    id: "section-attack-matrix",
    label: "攻击方法矩阵",
    shortLabel: "攻击矩阵",
    aliases: [],
    category: "section",
    sourcePages: ["dashboard", "glossary"],
    detailSections: [
      {
        title: "这块内容表示什么",
        paragraphs: [
          "攻击方法矩阵指平台当前已经接入并能在统一入口下调用的攻击方法集合。它不是简单列表，而是用类别、扰动形式、作用模态和实现方式组织起来的对照面板。",
          "展示这块内容，是为了回应学长强调的‘工具完整度’，证明平台既有论文主方法，也补齐了基础基线。",
        ],
      },
    ],
  }),
  makeEntry("section", "model-matrix", {
    id: "section-model-matrix",
    label: "受测模型矩阵",
    shortLabel: "模型矩阵",
    aliases: [],
    category: "section",
    sourcePages: ["dashboard", "glossary"],
    detailSections: [
      {
        title: "这块内容表示什么",
        paragraphs: [
          "受测模型矩阵指当前平台已经纳入统一评测链路的模型集合，页面重点展示模型名称、健康状态和轻量迁移验证覆盖情况。",
          "这里强调的是‘能否真实运行’，不是页面挂上 10 个名字就算完成。",
        ],
      },
    ],
  }),
  makeEntry("section", "dataset-matrix", {
    id: "section-dataset-matrix",
    label: "数据集矩阵",
    shortLabel: "数据集矩阵",
    aliases: [],
    category: "section",
    sourcePages: ["dashboard", "glossary"],
    detailSections: [
      {
        title: "这块内容表示什么",
        paragraphs: [
          "数据集矩阵指当前平台固定展示的真实数据集集合：图文检索使用 COCO/Flickr 图文检索数据，视觉问答和图像描述使用 COCO/VQA v2 生成式标注数据，用来说明平台能覆盖检索式和生成式两类测评场景。",
        ],
      },
    ],
  }),
  makeEntry("section", "latest-results", {
    id: "section-latest-results",
    label: "最近验证结果",
    shortLabel: "最近结果",
    aliases: ["最新结果"],
    category: "section",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [
      {
        title: "这个模块展示什么",
        paragraphs: [
          "它展示最近已经成功落盘的运行结果，而不是还在排队或运行中的任务。",
          "每条结果通常包含运行编号、攻击方法、受测模型、数据集和攻击成功率，用于证明系统的后台是真实在跑的。",
        ],
      },
      {
        title: "它与任务进度的区别",
        paragraphs: [
          "任务进度负责展示还没跑完的任务状态；最近验证结果展示的是已经完成并生成结果文件的运行记录。两者不能混为一谈。",
        ],
      },
    ],
    relatedIds: ["workflow-job-progress", "metric-asr"],
  }),
  makeEntry("workflow", "job-progress", {
    id: "workflow-job-progress",
    label: "当前任务进度",
    shortLabel: "任务进度",
    aliases: ["任务反馈", "任务进度卡"],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "它具体负责什么",
        paragraphs: [
          "当前任务进度卡负责把一次对抗测试从提交到完成的全过程展示出来，包括排队状态、当前阶段、已耗时、预计剩余时间、日志和最终运行编号。",
          "它的存在就是为了解决‘点击启动试跑以后没有任何反馈’的问题，让用户明确知道任务到底有没有真的在跑、跑到哪一步、还要等多久。",
        ],
      },
    ],
    relatedIds: ["status-queued", "status-running", "status-success", "status-failed"],
  }),
  makeEntry("workflow", "eval-type", {
    id: "workflow-eval-type",
    label: "运行方式",
    shortLabel: "运行方式",
    aliases: [],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个字段在页面里指什么",
        paragraphs: [
          "运行方式用于说明当前任务是在做交互式对抗试跑。它不是装饰性标题，而是决定整条运行链路和最终产物结构的入口字段。",
          "用户之所以必须看懂这个词，是因为同样的攻击方法、模型和数据集，在不同运行方式下生成的阶段结果、图表和结论并不相同。",
        ],
      },
    ],
    relatedIds: ["mode-standard-eval"],
  }),
  makeEntry("workflow", "experiment-id", {
    id: "workflow-experiment-id",
    label: "实验编号",
    shortLabel: "实验编号",
    aliases: ["experiment_id"],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个字段在系统里有什么用",
        paragraphs: [
          "实验编号是用户主动给一次测试附加的业务标识，用来把本次运行与答辩演示、论文实验批次或老师要求的某个固定实验对应起来。",
          "它不会改变算法结果本身，但会进入任务记录和结果落盘信息，方便后续筛选、核对和复述。",
        ],
      },
    ],
  }),
  makeEntry("workflow", "task-launch", {
    id: "workflow-task-launch",
    label: "启动试跑",
    shortLabel: "启动试跑",
    aliases: ["启动测试"],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个动作具体触发了什么",
        paragraphs: [
          "启动试跑会把当前页面中已经选定的攻击方法、模型、数据集和参数组合成一个真实的后端任务，并写入任务队列。",
          "从这一刻开始，页面不应该再只是按钮加载状态，而应该持续展示任务进度、预计剩余时间、日志和最终运行编号。这也是这次重做任务反馈链的核心原因。",
        ],
      },
    ],
    relatedIds: ["workflow-job-progress", "stage-queue"],
  }),
  makeEntry("workflow", "surrogate-model", {
    id: "workflow-surrogate-model",
    label: "代理模型",
    shortLabel: "代理模型",
    aliases: [],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个词在系统里指什么",
        paragraphs: [
          "代理模型是攻击优化时直接访问和计算梯度的模型，也是很多对抗攻击内部真正用来生成扰动的对象。",
          "平台把它和受测模型分开展示，是为了说明当前任务在做迁移攻击评测，而不是只攻击自己本身。",
        ],
      },
    ],
    relatedIds: ["workflow-victim-model", "model-clip"],
  }),
  makeEntry("workflow", "current-stage", {
    id: "workflow-current-stage",
    label: "当前阶段",
    shortLabel: "当前阶段",
    aliases: [],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个字段在任务反馈里表示什么",
        paragraphs: [
          "当前阶段表示后端任务此刻正在执行哪一个标准化步骤，例如排队中、模型预检查、数据集装载、执行攻击或结果汇总。",
          "它的作用是把原本看不见的后台流程显式展开，让用户知道任务卡在哪里，而不是只能盯着按钮等。",
        ],
      },
    ],
    relatedIds: ["workflow-job-progress", "stage-queue", "stage-attack-execution", "stage-report-writing"],
  }),
  makeEntry("workflow", "queue-position", {
    id: "workflow-queue-position",
    label: "队列位置",
    shortLabel: "队列位置",
    aliases: [],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个字段在任务反馈里表示什么",
        paragraphs: [
          "队列位置表示当前任务在后端等待执行队列中的相对顺序。数值越靠前，意味着越快轮到本任务实际开始执行。",
          "它用于回答『为什么我已经点击启动试跑，但结果还没出现在最近验证结果里』这个问题，因为任务可能还处于排队阶段。",
        ],
      },
    ],
    relatedIds: ["stage-queue", "workflow-job-progress"],
  }),
  makeEntry("workflow", "run-id", {
    id: "workflow-run-id",
    label: "运行编号",
    shortLabel: "运行编号",
    aliases: ["run_id"],
    category: "workflow",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [
      {
        title: "这个字段在系统里指什么",
        paragraphs: [
          "运行编号是一次真实实验落盘后的唯一标识，用来关联 摘要、报告数据、案例包等结果文件。",
          "它不是任务提交前的临时编号，而是任务成功完成后，平台可以长期复查和回放的结果编号。",
        ],
      },
    ],
    relatedIds: ["section-latest-results", "workflow-job-progress"],
  }),
  makeEntry("workflow", "latest-log", {
    id: "workflow-latest-log",
    label: "最近日志",
    shortLabel: "最近日志",
    aliases: ["最新日志"],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个字段在任务反馈里表示什么",
        paragraphs: [
          "最近日志显示的是后端任务最近输出的结构化日志片段，用来补充当前阶段之外的更细粒度提示。",
          "当任务失败时，它通常是定位问题最直接的线索；当任务成功时，它也能证明后台确实在逐步推进，而不是前端假进度。",
        ],
      },
    ],
    relatedIds: ["workflow-job-progress"],
  }),
  makeEntry("workflow", "victim-model", {
    id: "workflow-victim-model",
    label: "受测模型",
    shortLabel: "受测模型",
    aliases: ["受测模型列表", "受测模型数量"],
    category: "workflow",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [
      {
        title: "这个词在系统里指什么",
        paragraphs: [
          "受测模型是平台最终拿来验证攻击效果的对象。攻击后的输入会被送到这些模型上，再统计检索指标或判定结果。",
          "受测模型数量越多，越能证明平台是一个跨模型测评工具，而不是单模型演示。",
        ],
      },
    ],
    relatedIds: ["metric-validated-model-count"],
  }),
  makeEntry("mode", "standard-eval", {
    id: "mode-standard-eval",
    label: "交互式对抗试跑",
    shortLabel: "对抗试跑",
    aliases: [],
    category: "mode",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这种运行方式是什么意思",
        paragraphs: [
          "交互式对抗试跑只比较正常输入和攻击后输入，不额外加入其他处理。它适合现场观察某个攻击本身的破坏能力，也适合做小样本流程检查。",
          "这类试跑默认不会自动计入正式基准归档；即使手动选了正式基准集，页面也只会按当前设置的样本对数执行。",
        ],
      },
    ],
  }),
  makeEntry("parameter", "epsilon", {
    id: "param-epsilon",
    label: "扰动上限",
    shortLabel: "扰动上限",
    aliases: ["epsilon"],
    category: "parameter",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个参数控制什么",
        paragraphs: [
          "扰动上限限制对抗扰动的最大幅度，也就是限制攻击能把输入改到多大。在很多梯度攻击中，它对应的是 \\(\\epsilon\\) 预算。",
          "预算越大，攻击通常越容易成功，但输入改动也可能越明显；预算越小，攻击更克制，但成功率可能下降。",
        ],
      },
    ],
  }),
  makeEntry("section", "parameter-settings", {
    id: "section-parameter-settings",
    label: "参数设置",
    shortLabel: "参数设置",
    aliases: [],
    category: "section",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个模块负责什么",
        paragraphs: [
          "参数设置模块负责展示当前攻击方法真正需要的参数，而不是把所有算法的字段混在一起堆给用户。",
          "它的核心目的，是让用户理解不同攻击方法的参数语义不同，例如 AdvCLIP 的补丁尺寸和 FGSM 的步长并不是同一类控制量。",
        ],
      },
    ],
    relatedIds: ["param-epsilon", "param-step-size", "param-steps", "param-patch-size", "param-text-budget", "param-max-pairs"],
  }),
  makeEntry("section", "workload-check", {
    id: "section-workload-check",
    label: "工作量检查",
    shortLabel: "工作量检查",
    aliases: [],
    category: "section",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个模块为什么会出现在页面里",
        paragraphs: [
          "工作量检查模块用来把学长明确点过的数量要求直接映射成页面上的可见核对项，例如攻击方法数量、已验证模型数量和数据集数量。",
          "它不是学术指标，而是项目完整度检查项，目的是避免答辩时只能口头说『我大概够了』却没有现场可核对的依据。",
        ],
      },
    ],
    relatedIds: ["section-attack-matrix", "metric-validated-model-count", "section-dataset-matrix"],
  }),
  makeEntry("parameter", "step-size", {
    id: "param-step-size",
    label: "步长",
    shortLabel: "步长",
    aliases: ["step_size"],
    category: "parameter",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个参数控制什么",
        paragraphs: [
          "步长控制每一次迭代更新时，扰动朝当前梯度方向迈出多大的步子。它主要服务于多步迭代型攻击。",
        ],
      },
    ],
  }),
  makeEntry("parameter", "steps", {
    id: "param-steps",
    label: "迭代步数",
    shortLabel: "迭代步数",
    aliases: ["steps"],
    category: "parameter",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个参数控制什么",
        paragraphs: [
          "迭代步数表示攻击会重复优化多少轮。步数越多，通常优化越充分，但运行时间也更长。",
        ],
      },
    ],
  }),
  makeEntry("parameter", "patch-size", {
    id: "param-patch-size",
    label: "补丁尺寸",
    shortLabel: "补丁尺寸",
    aliases: ["patch_size"],
    category: "parameter",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个参数控制什么",
        paragraphs: [
          "补丁尺寸控制对抗补丁的空间大小，主要服务于 AdvCLIP 这类补丁式方法。尺寸越大，覆盖区域越多，攻击能力可能更强，但更容易被肉眼察觉。",
        ],
      },
    ],
  }),
  makeEntry("parameter", "text-budget", {
    id: "param-text-budget",
    label: "文本修改预算",
    shortLabel: "文本预算",
    aliases: ["eps_t"],
    category: "parameter",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个参数控制什么",
        paragraphs: [
          "文本修改预算限制图文联合攻击中，文本分支可以改动到什么程度。它只对涉及文本分支的方法有意义，因此页面会按方法动态显示。",
        ],
      },
    ],
  }),
  makeEntry("parameter", "max-pairs", {
    id: "param-max-pairs",
    label: "最大样本对数",
    shortLabel: "最大样本对数",
    aliases: ["max_pairs"],
    category: "parameter",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个参数控制什么",
        paragraphs: [
          "最大样本对数用于限制当前任务实际参与评测的图文配对规模。答辩演示或轻量迁移验证时，经常会把它调小，以便更快看到真实结果。",
        ],
      },
    ],
  }),
  makeEntry("metric", "validated-model-count", {
    id: "metric-validated-model-count",
    label: "已验证模型",
    shortLabel: "已验证模型",
    aliases: ["轻量迁移验证", "迁移验证"],
    category: "metric",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [
      {
        title: "这个指标表示什么",
        paragraphs: [
          "已验证模型数量表示已经通过轻量迁移验证的模型数量，而不是页面上仅仅挂出来的模型名字数量。",
          "这里的轻量迁移验证，指模型至少在 Flickr1k 切片和基准攻击集合上产生过一次可观测的对抗信号，用来证明模型接入后的真实可测性。",
        ],
      },
    ],
  }),
  makeEntry("metric", "elapsed-time", {
    id: "metric-elapsed-time",
    label: "已耗时",
    shortLabel: "已耗时",
    aliases: [],
    category: "metric",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个指标表示什么",
        paragraphs: [
          "已耗时表示当前任务从提交成功开始，已经过去了多少实际时间。它是任务反馈中最直观的进度时间量。",
          "它不能单独说明任务是否卡死，但和当前阶段、预计剩余时间一起看时，可以帮助用户判断任务是否按正常节奏推进。",
        ],
      },
    ],
    formulaBlocks: [
      {
        title: "已耗时计算",
        latex: "t_{elapsed}=t_{now}-t_{submit}",
        explanation: ["其中 \\(t_{submit}\\) 表示任务提交时间，\\(t_{now}\\) 表示当前时间。"],
        variables: [
          { symbol: "t_{elapsed}", meaning: "已耗时" },
          { symbol: "t_{now}", meaning: "当前时间" },
          { symbol: "t_{submit}", meaning: "任务提交时间" },
        ],
      },
    ],
  }),
  makeEntry("metric", "eta-time", {
    id: "metric-eta-time",
    label: "预计剩余时间",
    shortLabel: "预计剩余时间",
    aliases: ["ETA"],
    category: "metric",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个指标表示什么",
        paragraphs: [
          "预计剩余时间表示在当前阶段和历史同类任务时长估计下，本次任务距离完成大约还需要多久。",
          "它是估计值而不是保证值，因此更适合帮助用户建立等待预期，而不是当作严格截止时间。",
        ],
      },
    ],
    formulaBlocks: [
      {
        title: "预计剩余时间估计",
        latex: "t_{eta}=\\max(0,\\tilde{t}_{job}-t_{elapsed})",
        explanation: ["其中 \\(\\tilde{t}_{job}\\) 表示同类任务的估计总时长中位数。"],
        variables: [
          { symbol: "t_{eta}", meaning: "预计剩余时间" },
          { symbol: "\\tilde{t}_{job}", meaning: "历史同类任务的估计总时长" },
          { symbol: "t_{elapsed}", meaning: "当前已耗时" },
        ],
      },
    ],
  }),
  makeEntry("metric", "estimated-ready-at", {
    id: "metric-estimated-ready-at",
    label: "预计完成时间",
    shortLabel: "预计完成时间",
    aliases: [],
    category: "metric",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个指标表示什么",
        paragraphs: [
          "预计完成时间表示系统把预计剩余时间换算成具体时刻后的结果，让用户不用自己心算还要再等多久。",
          "这个时间会随着任务阶段推进持续刷新，因此它是动态预测值，不是写死的固定承诺。",
        ],
      },
    ],
    formulaBlocks: [
      {
        title: "预计完成时间计算",
        latex: "t_{ready}=t_{now}+t_{eta}",
        explanation: ["只要剩余时间估计发生变化，预计完成时间也会跟着更新。"],
        variables: [
          { symbol: "t_{ready}", meaning: "预计完成时刻" },
          { symbol: "t_{now}", meaning: "当前时刻" },
          { symbol: "t_{eta}", meaning: "预计剩余时间" },
        ],
      },
    ],
  }),
  makeEntry("metric", "asr", {
    id: "metric-asr",
    label: "攻击成功率（ASR）",
    shortLabel: "攻击成功率（ASR）",
    aliases: ["攻击成功率"],
    category: "metric",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [
      {
        title: "这个指标在本系统里指什么",
        paragraphs: [
          "攻击成功率用来衡量攻击是否成功破坏了模型在当前任务中的有效结果。本系统按任务类型分别解释：图文检索看正确图文配对是否掉出前 K 个候选；视觉问答看原始输入阶段本来答对的样本在攻击后是否答错；图像描述看目标对象是否按攻击目标被删除或被新增。",
          "所以它不是一个固定分母的“40000 配对错误率”。在检索里分母是查询或样本；在视觉问答和图像描述里，分母是原始输入阶段有效样本或具备目标对象判定条件的样本。",
        ],
      },
      {
        title: "如何解读它",
        paragraphs: [
          "数值越大，表示攻击更容易破坏当前任务：图文检索中是正确配对掉出前 k 个候选，视觉问答中是原本答对的问题被答错，图像描述中是目标对象出现状态被攻击翻转。高攻击成功率仍然要结合扰动成本和语义保持一起看。",
        ],
      },
      {
        title: "常见误解",
        paragraphs: [
          "不能把攻击成功率简单理解成所有任务统一的‘模型输出完全错了的比例’。图文检索、视觉问答和图像描述的输出形态不同，因此攻击成功条件也必须分别定义。",
        ],
      },
    ],
    formulaBlocks: [
      {
        title: "图文检索攻击成功率",
        latex: String.raw`\mathrm{ASR}_{\mathrm{VLR}, k}=1-\mathrm{Recall}_{k}`,
        explanation: ["如果某个查询的正确图文配对不再留在前 k 个检索候选里，这个查询就记为攻击成功。"],
        variables: [
          { symbol: String.raw`\mathrm{ASR}_{\mathrm{VLR}, k}`, meaning: "图文检索任务在前 K 个候选条件下的攻击成功率。" },
          { symbol: String.raw`\mathrm{Recall}_{k}`, meaning: "正确配对仍然保留在前 k 个候选内的召回率。" },
        ],
      },
      {
        title: "视觉问答攻击成功率",
        latex: String.raw`\mathrm{ASR}_{\mathrm{VQA}}=\frac{\#\{\mathrm{clean\ correct}\land\mathrm{attacked\ wrong}\}}{\#\{\mathrm{clean\ correct}\}}`,
        explanation: ["视觉问答只在原图阶段本来答对的样本上统计攻击成功，避免把模型原始错误也算成攻击效果。"],
        variables: [
          { symbol: String.raw`\mathrm{clean\ correct}`, meaning: "模型在原始图片和同一个问题上回答正确。" },
          { symbol: String.raw`\mathrm{attacked\ wrong}`, meaning: "模型在攻击后图片和同一个问题上回答错误。" },
        ],
      },
      {
        title: "图像描述目标对象翻转率",
        latex: String.raw`\mathrm{ASR}_{\mathrm{Caption}}=\frac{\#\{\mathrm{target\ flip\ success}\}}{N_{\mathrm{valid}}}`,
        explanation: ["图像描述任务把攻击成功落到对象级语义：AdvEDM-R 类目标是让原本存在的目标对象在攻击后描述中消失，AdvEDM-A 类目标是让原本不存在的目标对象在攻击后描述中出现。"],
        variables: [
          { symbol: String.raw`\mathrm{target\ flip\ success}`, meaning: "目标对象的出现状态按攻击目标发生翻转。" },
          { symbol: String.raw`N_{\mathrm{valid}}`, meaning: "具备目标对象判定条件的有效图像描述样本数。" },
        ],
      },
    ],
    chartExplainBlocks: [
      {
        title: "相关图表怎么读",
        items: [
          { label: "指标总览图", detail: "先看攻击成功率、风险分数和扰动指标，抓住这次运行的整体结论。" },
          { label: "阶段对比图", detail: "比较正常输入和攻击后输入，直观看攻击破坏效果。" },
          { label: "模型差异图", detail: "看哪些受测模型的攻击成功率更高，回答‘攻击具体打到了哪些模型’。" },
          { label: "样本分布图", detail: "看成功样本和失败样本如何分布，避免只看均值掩盖细节。" },
        ],
      },
    ],
    relatedIds: ["chart-stage-compare", "chart-sample-distribution"],
  }),
  makeEntry("metric", "risk-score", {
    id: "metric-risk-score",
    label: "平台综合风险分数",
    shortLabel: "风险分数",
    aliases: ["风险分数"],
    category: "metric",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个指标在本系统里指什么",
        paragraphs: [
          "平台综合风险分数是系统内部定义的排序指标，用来综合攻击有效性、语义保持、代价、跨受测模型迁移覆盖情况和稳定性，并将这些因素汇总为 0 到 1 之间的分数。",
          "它的作用是帮助你快速排序不同运行结果的风险高低，而不是代替论文里的通用学术指标。",
        ],
      },
      {
        title: "必须注意的边界",
        paragraphs: [
          "这个分数是平台内部综合指标，不应表述为国际通用标准，也不应用来单独证明某个攻击方法在论文层面成立。更准确的说法是：平台为便于比较不同运行结果，自定义了一个综合风险分数。",
        ],
      },
    ],
    formulaBlocks: [
      {
        title: "综合风险分数定义",
        latex: String.raw`\mathrm{RiskScore}=\sum_{d \in D} w_d \cdot s_d`,
        explanation: ["每个风险维度都会先归一化到 0 到 1，再乘以该场景下的权重，最后加权求和。"],
        variables: [
          { symbol: String.raw`D`, meaning: "风险维度集合，包括攻击有效性、语义保持、扰动代价、迁移性和稳定性。" },
          { symbol: String.raw`w_d`, meaning: "第 d 个风险维度的权重。" },
          { symbol: String.raw`s_d`, meaning: "第 d 个风险维度归一化后的得分。" },
        ],
      },
    ],
    relatedIds: ["chart-risk-summary"],
  }),
  makeEntry("status", "online", {
    id: "status-online",
    label: "在线",
    shortLabel: "在线",
    aliases: [],
    category: "status",
    sourcePages: ["layout", "glossary"],
    detailSections: [{ title: "这个状态说明什么", paragraphs: ["在线表示当前前端已经连上后端，并且页面处于可交互状态。它不代表某个攻击任务已经完成。"] }],
  }),
  makeEntry("status", "ready", {
    id: "status-ready",
    label: "就绪",
    shortLabel: "就绪",
    aliases: [],
    category: "status",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [{ title: "这个状态说明什么", paragraphs: ["就绪表示当前模型或组件已经通过预检查，可以直接进入真实任务链路。"] }],
  }),
  makeEntry("status", "launchable", {
    id: "status-launchable",
    label: "已发现启动脚本",
    shortLabel: "已发现脚本",
    aliases: ["可启动"],
    category: "status",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [
      {
        title: "这个状态说明什么",
        paragraphs: [
          "这个状态表示模型当前还未开始运行，但平台已经找到了对应的启动脚本。",
          "这并不表示模型权重、本地文件、依赖环境、端口和显存都已准备好；模型最终能否启动，还要看后续预检查和健康检查结果。",
        ],
      },
    ],
  }),
  makeEntry("status", "launch_blocked", {
    id: "status-launch-blocked",
    label: "当前环境不可启动",
    shortLabel: "不可启动",
    aliases: ["启动受阻"],
    category: "status",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [
      {
        title: "这个状态说明什么",
        paragraphs: [
          "这表示平台已经识别出对应模型适配器，但当前机器上的启动前提还不满足，比如启动脚本、Python 运行时、可写目录或显卡选择仍有问题。",
          "遇到这个状态时，测试页不应再把它当成可直接启动的在线受测模型。",
        ],
      },
    ],
  }),
  makeEntry("status", "queued", {
    id: "status-queued",
    label: "排队中",
    shortLabel: "排队中",
    aliases: [],
    category: "status",
    sourcePages: ["testing", "glossary"],
    detailSections: [{ title: "这个状态说明什么", paragraphs: ["排队中表示任务已经成功进入后端队列，但还没有轮到后端执行进程真正开始执行。"] }],
  }),
  makeEntry("status", "running", {
    id: "status-running",
    label: "运行中",
    shortLabel: "运行中",
    aliases: [],
    category: "status",
    sourcePages: ["testing", "glossary"],
    detailSections: [{ title: "这个状态说明什么", paragraphs: ["运行中表示任务已经被后端执行进程接手，正在执行模型预检查、攻击生成、受测模型评测或结果写盘中的某个阶段。"] }],
  }),
  makeEntry("status", "success", {
    id: "status-success",
    label: "成功",
    shortLabel: "成功",
    aliases: [],
    category: "status",
    sourcePages: ["testing", "glossary"],
    detailSections: [{ title: "这个状态说明什么", paragraphs: ["成功表示任务已经完整执行完毕，并且结果文件已经写入运行目录，因此会出现在最近验证结果中。"] }],
  }),
  makeEntry("status", "failed", {
    id: "status-failed",
    label: "失败",
    shortLabel: "失败",
    aliases: [],
    category: "status",
    sourcePages: ["testing", "glossary"],
    detailSections: [{ title: "这个状态说明什么", paragraphs: ["失败表示任务在执行链路中出现错误，例如模型未就绪、配置错误、数据集缺失或运行阶段异常退出。"] }],
  }),
  makeEntry("status", "local-model", {
    id: "status-local-model",
    label: "本地模型",
    shortLabel: "本地模型",
    aliases: [],
    category: "status",
    sourcePages: ["layout", "dashboard", "testing", "glossary"],
    detailSections: [{ title: "这个词表示什么", paragraphs: ["本地模型指模型权重和推理过程部署在当前服务器环境中，不依赖云端闭源接口即可执行。"] }],
  }),
  makeEntry("status", "service-model", {
    id: "status-service-model",
    label: "接口模型",
    shortLabel: "接口模型",
    aliases: [],
    category: "status",
    sourcePages: ["layout", "dashboard", "testing", "glossary"],
    detailSections: [{ title: "这个词表示什么", paragraphs: ["接口模型指通过兼容接口或本地自托管服务提供推理能力的模型。在当前项目里，它主要指本地自托管视觉-语言模型服务。"] }],
  }),
  makeEntry("status", "local-load", {
    id: "status-local-load",
    label: "本地加载",
    shortLabel: "本地加载",
    aliases: [],
    category: "status",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [{ title: "这个接入方式表示什么", paragraphs: ["本地加载表示后端直接在本机 Python 进程或同机环境中加载模型权重并做推理，不需要额外 HTTP 服务层。"] }],
  }),
  makeEntry("status", "api-access", {
    id: "status-api-access",
    label: "接口接入",
    shortLabel: "接口接入",
    aliases: [],
    category: "status",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [{ title: "这个接入方式表示什么", paragraphs: ["接口接入表示平台通过兼容接口调用模型服务，服务本身可能是本地自托管，也可能是外部接口。"] }],
  }),
  makeEntry("chart", "metric-overview", {
    id: "chart-metric-overview",
    label: "指标总览图",
    shortLabel: "指标总览图",
    aliases: [],
    category: "chart",
    sourcePages: ["testing", "glossary"],
    detailSections: [{ title: "这张图回答什么问题", paragraphs: ["它把攻击成功率、平台综合风险分数和扰动指标放在一起，帮助用户先抓住这次运行的整体结论。"] }],
  }),
  makeEntry("chart", "stage-compare", {
    id: "chart-stage-compare",
    label: "阶段对比图",
    shortLabel: "阶段对比图",
    aliases: [],
    category: "chart",
    sourcePages: ["testing", "glossary"],
    detailSections: [{ title: "这张图回答什么问题", paragraphs: ["阶段对比图用于展示正常输入和攻击后输入的指标变化，让人直观看到攻击破坏效果。"] }],
  }),
  makeEntry("chart", "model-difference", {
    id: "chart-model-difference",
    label: "模型差异图",
    shortLabel: "模型差异图",
    aliases: [],
    category: "chart",
    sourcePages: ["testing", "glossary"],
    detailSections: [{ title: "这张图回答什么问题", paragraphs: ["它用来比较同一次攻击在不同受测模型上的效果差异，回答‘攻击到底对哪些模型更有效’。"] }],
  }),
  makeEntry("chart", "sample-distribution", {
    id: "chart-sample-distribution",
    label: "样本分布图",
    shortLabel: "样本分布图",
    aliases: [],
    category: "chart",
    sourcePages: ["testing", "glossary"],
    detailSections: [{ title: "这张图回答什么问题", paragraphs: ["它说明攻击结果不是只体现在一个平均值上，而是落在每个样本上的真实分布。"] }],
  }),
  makeEntry("chart", "risk-summary", {
    id: "chart-risk-summary",
    label: "风险分解图",
    shortLabel: "风险分解图",
    aliases: ["风险贡献图"],
    category: "chart",
    sourcePages: ["testing", "glossary"],
    detailSections: [{ title: "这张图回答什么问题", paragraphs: ["它把平台综合风险分数拆成多个维度，回答‘为什么这个运行的总风险高’。"] }],
  }),
];

const attackDetailExtras: Record<string, string[]> = {
  advclip: ["它属于补丁式攻击方法，需要先训练或载入通用补丁，再拿补丁做评测，因此参数语义和普通梯度噪声攻击不同。"],
  tmm: ["它同时改动图像和文本，因此属于图文联合攻击方法。"],
  advedm_plus: ["它是在 AdvEDM 基础上引入文本分支和自适应预算的增强版，是当前项目里最接近改进方法定位的一条路线。"],
  fgsm: ["它是最简单的单步梯度攻击，经常被当作基础白盒基线。"],
  pgd: ["它是最常用的强基线之一，经常被用来验证一个系统是否真正支持标准梯度攻击。"],
  cw: ["它是经典优化型攻击方法，和梯度符号法属于不同优化风格。"],
};

const attackEntries: GlossaryEntry[] = attackCatalog.map((item) =>
  makeEntry("attack", item.id.replace(/_/g, "-"), {
    id: item.glossaryId,
    label: item.name,
    shortLabel: item.name,
    aliases: [item.id, item.name],
    category: "attack",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [
      {
        title: "这个方法在本系统里指什么",
        paragraphs: [
          `${item.name} 是平台攻击方法矩阵中的一条攻击路线，归类为“${item.category}”。`,
          `它的扰动形式是“${item.perturbation}”，作用模态是“${item.modality}”，实现方式是“${item.training}”。`,
        ],
      },
      {
        title: "具体功能与展示原因",
        paragraphs: [
          item.summary,
          "页面展示它，是为了证明这条攻击路线已经纳入统一评测入口，而不是只在代码里有一个占位名字。",
          ...(attackDetailExtras[item.id] ?? []),
        ],
      },
      {
        title: "如何解读它",
        paragraphs: [
          "如果它是论文主方法或改进方法，答辩时重点在于解释方法思路和它在平台里的代表性。",
          "如果它是基础基线，重点是说明平台覆盖面完整，能够补齐最常见的标准攻击方法。",
        ],
      },
    ],
    relatedIds: ["section-attack-matrix", "param-epsilon", "param-steps"],
  }),
);

const modelEntries: GlossaryEntry[] = recommendedModels.map((item) =>
  makeEntry("model", item.id, {
    id: item.glossaryId,
    label: item.name,
    shortLabel: item.name,
    aliases: [item.adapter, item.id, item.name],
    category: "model",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [
      {
        title: "这个模型在本系统里指什么",
        paragraphs: [
          `${item.name} 是平台受测模型矩阵中的一个具体模型条目，接入适配器标识为 ${item.adapter}。`,
          `它当前被归为“${item.family === "local" ? "本地模型" : "接口模型"}”，部署方式是“${item.deployment}”。`,
        ],
      },
      {
        title: "它的具体功能是什么",
        paragraphs: [item.summary, "它出现在页面里，是为了证明平台不是只测一两个模型，而是具备跨模型评测能力。"],
      },
    ],
    relatedIds: ["workflow-victim-model", "metric-validated-model-count"],
  }),
);

const datasetEntries: GlossaryEntry[] = datasetCatalog.map((item) =>
  makeEntry("dataset", item.id.replace(/_/g, "-"), {
    id: item.glossaryId,
    label: item.title,
    shortLabel: item.title,
    aliases: [item.id, item.shortName, item.title],
    category: "dataset",
    sourcePages: ["dashboard", "testing", "glossary"],
    detailSections: [
      {
        title: "这个数据集在本系统里指什么",
        paragraphs: [
          `${item.title} 是当前平台纳入答辩展示的核心数据集条目。`,
          item.summary,
          item.tier === "generation"
            ? "它服务 视觉问答或图像描述生成式评测，不参与图文检索的 N×N 图文检索矩阵；运行时必须由真实生成式视觉语言模型输出答案或描述。"
            : "它服务图文检索矩阵评测，用于构造图像集合和文本集合之间的相似度候选空间。",
        ],
      },
      {
        title: "为什么要用它",
        paragraphs: [
          item.tier === "generation"
            ? "生成式扩展任务需要真实公开标注来判断答案正确性、目标对象是否出现和非目标语义是否保持，因此使用 COCO/VQA v2 这类公开数据而不是临时编造样本。"
            : "平台同时保留主实验数据集、跨数据集验证数据集和轻量演示数据集，是为了在完整度、泛化能力和现场演示速度之间取得平衡。",
        ],
      },
    ],
    relatedIds: ["section-dataset-matrix"],
  }),
);

const progressStageEntries: GlossaryEntry[] = [
  makeEntry("workflow", "stage-queue", {
    id: "stage-queue",
    label: "排队中",
    shortLabel: "排队中",
    aliases: ["队列位置"],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个阶段在做什么",
        paragraphs: [
          "排队中表示任务已经成功写入后端任务表，但当前后端执行进程还没有真正开始执行这条任务。",
          "这个阶段最重要的信息是队列位置和预计等待时间，它回答的是“任务有没有真的提交成功、还要排多久”。",
        ],
      },
    ],
    relatedIds: ["workflow-job-progress", "status-queued"],
  }),
  makeEntry("workflow", "stage-model-preflight", {
    id: "stage-model-preflight",
    label: "模型预检查",
    shortLabel: "模型预检查",
    aliases: [],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个阶段在做什么",
        paragraphs: [
          "模型预检查用于确认当前任务依赖的代理模型和受测模型是否真的可用，包括本地权重是否存在、服务端口是否可连、接口是否能返回正常响应。",
          "把这个阶段单独展示出来，是为了避免任务失败时只看到一个笼统的报错，而不知道问题其实出在模型没有就绪。",
        ],
      },
    ],
  }),
  makeEntry("section", "result-insights", {
    id: "section-result-insights",
    label: "结果解读区",
    shortLabel: "结果解读区",
    aliases: ["结果图表区"],
    category: "section",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个模块负责什么",
        paragraphs: [
          "结果解读区负责把已经完成的运行结果从一个抽象指标表，转成用户能直观看懂的图表解释区。",
          "它的目标不是重复展示数字，而是回答『攻击到底怎么影响了结果』『哪些模型更脆弱』『哪些样本风险更高』这些具体问题。",
        ],
      },
    ],
    relatedIds: ["chart-metric-overview", "chart-stage-compare", "chart-model-difference", "chart-sample-distribution"],
  }),
  makeEntry("workflow", "stage-config-validation", {
    id: "stage-config-validation",
    label: "配置校验",
    shortLabel: "配置校验",
    aliases: [],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个阶段在做什么",
        paragraphs: [
          "配置校验会检查攻击方法、模型、数据集和参数组合是否满足当前任务类型的要求，例如某些方法是否允许文本预算、某些任务是否需要补丁尺寸。",
          "这个阶段出错通常不是算法失败，而是任务参数本身不合法。",
        ],
      },
    ],
  }),
  makeEntry("workflow", "stage-dataset-loading", {
    id: "stage-dataset-loading",
    label: "数据集装载",
    shortLabel: "数据集装载",
    aliases: [],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个阶段在做什么",
        paragraphs: [
          "数据集装载阶段负责读取当前任务要使用的图文样本，并整理成攻击与评测所需的统一输入结构。",
          "如果这个阶段失败，通常意味着数据目录、索引文件或轻量演示数据集没有正确部署。",
        ],
      },
    ],
  }),
  makeEntry("workflow", "stage-attack-execution", {
    id: "stage-attack-execution",
    label: "执行攻击",
    shortLabel: "执行攻击",
    aliases: ["补丁训练"],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个阶段在做什么",
        paragraphs: [
          "执行攻击阶段是当前任务真正生成对抗扰动或对抗补丁的阶段，也是整条链路里最核心的算法阶段。",
          "不同方法在这里执行的内容不同：经典梯度攻击是在预算内优化扰动，AdvCLIP 则可能先载入或训练补丁后再应用到样本上。",
        ],
      },
    ],
  }),
  makeEntry("workflow", "stage-victim-evaluation", {
    id: "stage-victim-evaluation",
    label: "受测模型评测",
    shortLabel: "受测模型评测",
    aliases: [],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个阶段在做什么",
        paragraphs: [
          "受测模型评测阶段会把正常样本和攻击后样本送入受测模型，并计算检索或判定相关指标。",
          "这个阶段回答的是“攻击最终对哪些模型造成了多大破坏”，因此它直接决定攻击成功率等结果指标。",
        ],
      },
    ],
  }),
  makeEntry("workflow", "stage-result-aggregation", {
    id: "stage-result-aggregation",
    label: "结果汇总",
    shortLabel: "结果汇总",
    aliases: [],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个阶段在做什么",
        paragraphs: [
          "结果汇总阶段负责把样本级结果整理为运行级指标，例如攻击成功率、平台综合风险分数和扰动摘要。",
          "如果前面的算法已经执行完，这个阶段主要是在把原始结果转成可以展示、可以比较、可以写入报告的结构化摘要。",
        ],
      },
    ],
  }),
  makeEntry("workflow", "stage-report-writing", {
    id: "stage-report-writing",
    label: "报告写入",
    shortLabel: "报告写入",
    aliases: [],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个阶段在做什么",
        paragraphs: [
          "报告写入阶段会把 摘要、报告数据、案例包等结果文件落盘，并把运行编号写回平台缓存。",
          "只有这个阶段完成之后，结果才会出现在“最近验证结果”中，因此它是进度面板和结果列表之间的桥接阶段。",
        ],
      },
    ],
  }),
  makeEntry("workflow", "stage-completed", {
    id: "stage-completed",
    label: "完成",
    shortLabel: "完成",
    aliases: ["任务完成"],
    category: "workflow",
    sourcePages: ["testing", "glossary"],
    detailSections: [
      {
        title: "这个阶段在做什么",
        paragraphs: [
          "完成表示当前任务已经完整跑通，并且成功生成了可以查看的运行编号与结果文件。",
          "只有当任务进入完成状态后，才应把它当成真实可复核的实验结果，而不是还在中途的暂存状态。",
        ],
      },
    ],
  }),
];

/** 中文注释：实现 isCurrentTopicEntry 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
function isCurrentTopicEntry(entry: GlossaryEntry): boolean {
  const raw = JSON.stringify(entry).toLowerCase();
  const blocked = ["\u9632\u5fa1", "\u653b\u9632", "defense", "defended"];
  return blocked.every((term) => !raw.includes(term));
}

export const glossaryEntries: GlossaryEntry[] = [...staticEntries, ...attackEntries, ...modelEntries, ...datasetEntries, ...progressStageEntries].filter(isCurrentTopicEntry);
export const glossaryEntryMap = new Map(glossaryEntries.map((item) => [item.id, item]));

/** 中文注释：实现 getGlossaryEntry 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function getGlossaryEntry(id: string) {
  return glossaryEntryMap.get(id);
}

/** 中文注释：实现 glossaryHref 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function glossaryHref(id: string) {
  const entry = glossaryEntryMap.get(id);
  return entry ? `/glossary#${entry.anchor}` : "/glossary";
}

/** 中文注释：实现 glossarySearchText 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function glossarySearchText(entry: GlossaryEntry) {
  const detail = entry.detailSections.flatMap((item) => item.paragraphs).join(" ");
  const formulas = (entry.formulaBlocks ?? []).flatMap((item) => [item.title, item.latex, ...item.explanation]).join(" ");
  const chartInfo = (entry.chartExplainBlocks ?? []).flatMap((item) => [item.title, ...item.items.map((x) => `${x.label} ${x.detail}`)]).join(" ");
  return [entry.label, entry.shortLabel, ...entry.aliases, detail, formulas, chartInfo].join(" ").toLowerCase();
}

export const glossaryCategoryLabel: Record<GlossaryCategory, string> = {
  platform: "平台",
  page: "页面",
  section: "页面模块",
  workflow: "流程概念",
  attack: "攻击方法",
  model: "模型",
  dataset: "数据集",
  mode: "评测模式",
  parameter: "参数",
  metric: "指标",
  status: "状态",
  chart: "图表",
};

export const glossaryGroupOrder: GlossaryCategory[] = [
  "platform",
  "page",
  "section",
  "workflow",
  "attack",
  "model",
  "dataset",
  "mode",
  "parameter",
  "metric",
  "status",
  "chart",
];

export const healthStatusGlossaryIds: Record<string, string> = {
  ready: "status-ready",
  launchable: "status-launchable",
  launch_blocked: "status-launch-blocked",
};

export const jobStatusGlossaryIds: Record<string, string> = {
  queued: "status-queued",
  running: "status-running",
  success: "status-success",
  failed: "status-failed",
};

export const jobStageGlossaryIds: Record<string, string> = {
  queued: "stage-queue",
  model_preflight: "stage-model-preflight",
  config_validation: "stage-config-validation",
  dataset_loading: "stage-dataset-loading",
  attack_execution: "stage-attack-execution",
  victim_evaluation: "stage-victim-evaluation",
  result_aggregation: "stage-result-aggregation",
  report_writing: "stage-report-writing",
  completed: "stage-completed",
};

export const modeGlossaryIds: Record<string, string> = {
  standard: "mode-standard-eval",
};

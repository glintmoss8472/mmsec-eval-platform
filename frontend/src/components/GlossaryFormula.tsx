// 文件说明：该文件属于前端组件，集中实现 GlossaryFormula 相关逻辑。
import { BlockMath } from "react-katex";

import type { GlossaryFormulaBlock } from "../lib/glossaryRegistry";

type GlossaryFormulaProps = {
  block: GlossaryFormulaBlock;
};

/** 渲染 `GlossaryFormula` 组件，组织该区域的数据读取、交互状态和可访问性标记。 */
export function GlossaryFormula({ block }: GlossaryFormulaProps) {
  return (
    <div className="glossary-formula-block">
      <div className="glossary-subtitle">{block.title}</div>
      <div className="glossary-formula">
        <BlockMath math={block.latex} />
      </div>
      <div className="glossary-copy">
        {block.explanation.map((line) => (
          <p key={line}>{line}</p>
        ))}
      </div>
      {block.variables?.length ? (
        <div className="table-wrap glossary-variable-table mt-3">
          <table className="data-table">
            <thead>
              <tr>
                <th>符号</th>
                <th>含义</th>
              </tr>
            </thead>
            <tbody>
              {block.variables.map((item) => (
                <tr key={`${block.title}-${item.symbol}`}>
                  <td className="font-mono text-xs">{item.symbol}</td>
                  <td>{item.meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

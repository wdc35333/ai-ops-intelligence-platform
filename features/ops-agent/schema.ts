// 운영 에이전트의 구조화된 리포트 스키마.
// Zod(런타임 검증) + 동일 의미의 JSON Schema(submit_report 도구 입력)를 함께 둔다.
import { z } from 'zod';

export const SEVERITIES = ['info', 'warning', 'critical'] as const;
export type Severity = (typeof SEVERITIES)[number];

export const opsReportSchema = z.object({
  severity: z.enum(SEVERITIES),
  headline: z.string(),
  sections: z.array(z.object({ title: z.string(), detail: z.string() })),
  anomalies: z.array(z.object({ severity: z.enum(SEVERITIES), summary: z.string() })),
  recommendations: z.array(z.string()),
});

export type OpsReport = z.infer<typeof opsReportSchema>;

// submit_report 도구의 input_schema (위 Zod 와 의미 동기화).
export const opsReportJsonSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['severity', 'headline', 'sections', 'anomalies', 'recommendations'],
  properties: {
    severity: { type: 'string', enum: [...SEVERITIES], description: '전체 심각도' },
    headline: { type: 'string', description: '가장 중요한 한 줄 요약(한국어)' },
    sections: {
      type: 'array',
      description: '영역별 요약(백업/시스템/DB/매출/ML 이상치 등)',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'detail'],
        properties: { title: { type: 'string' }, detail: { type: 'string' } },
      },
    },
    anomalies: {
      type: 'array',
      description: '발견된 이상 목록',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'summary'],
        properties: {
          severity: { type: 'string', enum: [...SEVERITIES] },
          summary: { type: 'string' },
        },
      },
    },
    recommendations: {
      type: 'array',
      description: '사람이 취할 다음 행동(없으면 빈 배열)',
      items: { type: 'string' },
    },
  },
} as const;

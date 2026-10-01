import type { RequestLogPricingLineDto } from '@shared/control/resources/request-logs'
import { formatExactNanoUSD } from '@shared/lib/format'
import { formatLogTokenCount } from './log-format'

export function formatReceiptFormulaLine(line: RequestLogPricingLineDto, locale: string): string {
  const quantity = formatLogTokenCount(line.quantity, locale)
  if (line.state === 'unpriced' || line.rate_nano_usd_per_million === null) return `${quantity} × —`
  const coefficient =
    line.multiplier.numerator === line.multiplier.denominator
      ? ''
      : ` × ${line.multiplier.numerator}/${line.multiplier.denominator}`
  return `${quantity} × ${formatExactNanoUSD(line.rate_nano_usd_per_million, locale)}/1M${coefficient}`
}

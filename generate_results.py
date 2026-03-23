"""
Generate results tables and comparison with paper (Table 5).
"""
import json
import os

def main():
    results_file = './results/results.json'
    with open(results_file) as f:
        results = json.load(f)

    # Paper targets (Table 5 MSE values)
    targets_mse = {
        'ETTh1': {'96': 0.375, '192': 0.398, '336': 0.419, '720': 0.422},
        'ETTh2': {'96': 0.253, '192': 0.334, '336': 0.347, '720': 0.401},
        'ETTm1': {'96': 0.322, '192': 0.349, '336': 0.366, '720': 0.407},
        'ETTm2': {'96': 0.173, '192': 0.230, '336': 0.279, '720': 0.388},
        'electricity': {'96': 0.142, '192': 0.153, '336': 0.175, '720': 0.209},
        'exchange_rate': {'96': 0.163, '192': 0.229, '336': 0.383, '720': 0.999},
        'traffic': {'96': 0.396, '192': 0.408, '336': 0.427, '720': 0.449},
        'weather': {'96': 0.161, '192': 0.208, '336': 0.252, '720': 0.337},
    }

    # Paper targets (Table 5 MAE values)
    targets_mae = {
        'ETTh1': {'96': 0.395, '192': 0.413, '336': 0.424, '720': 0.440},
        'ETTh2': {'96': 0.325, '192': 0.374, '336': 0.389, '720': 0.425},
        'ETTm1': {'96': 0.357, '192': 0.375, '336': 0.387, '720': 0.412},
        'ETTm2': {'96': 0.261, '192': 0.302, '336': 0.332, '720': 0.400},
        'electricity': {'96': 0.237, '192': 0.249, '336': 0.268, '720': 0.300},
        'exchange_rate': {'96': 0.276, '192': 0.336, '336': 0.445, '720': 0.738},
        'traffic': {'96': 0.271, '192': 0.279, '336': 0.290, '720': 0.310},
        'weather': {'96': 0.207, '192': 0.250, '336': 0.289, '720': 0.348},
    }

    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'exchange_rate', 'traffic', 'weather']
    horizons = ['96', '192', '336', '720']

    # Generate text report
    lines = []
    lines.append('=' * 120)
    lines.append('TSM2 (MambaMixer) Replication Results - Table 5 Comparison')
    lines.append('=' * 120)
    lines.append('')
    
    # MSE Table
    lines.append('MSE Results (Ours / Paper):')
    lines.append('-' * 100)
    header = f'{"Dataset":<15}'
    for h in horizons:
        header += f'  {"H=" + h:>16}'
    header += f'  {"Avg Ratio":>10}'
    lines.append(header)
    lines.append('-' * 100)

    all_mse_ratios = []
    for ds in datasets:
        row = f'{ds:<15}'
        ratios = []
        for h in horizons:
            ours = results[ds][h]['mse']
            target = targets_mse[ds][h]
            ratio = ours / target
            ratios.append(ratio)
            all_mse_ratios.append(ratio)
            row += f'  {ours:.3f}/{target:.3f}'
        avg_ratio = sum(ratios) / len(ratios)
        row += f'  {avg_ratio:.3f}x'
        lines.append(row)

    lines.append('-' * 100)
    lines.append(f'Overall MSE ratio: {sum(all_mse_ratios)/len(all_mse_ratios):.3f}x')
    lines.append(f'Within 10%: {sum(1 for r in all_mse_ratios if r <= 1.10)}/{len(all_mse_ratios)}')
    lines.append(f'Within 15%: {sum(1 for r in all_mse_ratios if r <= 1.15)}/{len(all_mse_ratios)}')
    lines.append(f'Better than paper: {sum(1 for r in all_mse_ratios if r < 1.0)}/{len(all_mse_ratios)}')
    lines.append('')

    # MAE Table
    lines.append('MAE Results (Ours / Paper):')
    lines.append('-' * 100)
    header = f'{"Dataset":<15}'
    for h in horizons:
        header += f'  {"H=" + h:>16}'
    header += f'  {"Avg Ratio":>10}'
    lines.append(header)
    lines.append('-' * 100)

    all_mae_ratios = []
    for ds in datasets:
        row = f'{ds:<15}'
        ratios = []
        for h in horizons:
            ours = results[ds][h]['mae']
            target = targets_mae[ds][h]
            ratio = ours / target
            ratios.append(ratio)
            all_mae_ratios.append(ratio)
            row += f'  {ours:.3f}/{target:.3f}'
        avg_ratio = sum(ratios) / len(ratios)
        row += f'  {avg_ratio:.3f}x'
        lines.append(row)

    lines.append('-' * 100)
    lines.append(f'Overall MAE ratio: {sum(all_mae_ratios)/len(all_mae_ratios):.3f}x')
    lines.append(f'Within 10%: {sum(1 for r in all_mae_ratios if r <= 1.10)}/{len(all_mae_ratios)}')
    lines.append(f'Within 15%: {sum(1 for r in all_mae_ratios if r <= 1.15)}/{len(all_mae_ratios)}')
    lines.append(f'Better than paper: {sum(1 for r in all_mae_ratios if r < 1.0)}/{len(all_mae_ratios)}')
    lines.append('')

    # Summary
    lines.append('=' * 120)
    lines.append('SUMMARY')
    lines.append('=' * 120)
    lines.append(f'Total experiments: 32 (8 datasets × 4 horizons)')
    lines.append(f'MSE: avg ratio = {sum(all_mse_ratios)/len(all_mse_ratios):.3f}x, {sum(1 for r in all_mse_ratios if r <= 1.10)}/32 within 10%')
    lines.append(f'MAE: avg ratio = {sum(all_mae_ratios)/len(all_mae_ratios):.3f}x, {sum(1 for r in all_mae_ratios if r <= 1.10)}/32 within 10%')
    lines.append('')

    report = '\n'.join(lines)
    print(report)

    # Save report
    os.makedirs('./results', exist_ok=True)
    with open('./results/results_table.txt', 'w') as f:
        f.write(report)
    print(f'\nSaved to ./results/results_table.txt')

    # Also generate a CSV
    with open('./results/results.csv', 'w') as f:
        f.write('Dataset,Horizon,MSE_Ours,MSE_Paper,MSE_Ratio,MAE_Ours,MAE_Paper,MAE_Ratio\n')
        for ds in datasets:
            for h in horizons:
                mse_ours = results[ds][h]['mse']
                mse_paper = targets_mse[ds][h]
                mae_ours = results[ds][h]['mae']
                mae_paper = targets_mae[ds][h]
                f.write(f'{ds},{h},{mse_ours:.6f},{mse_paper:.3f},{mse_ours/mse_paper:.4f},{mae_ours:.6f},{mae_paper:.3f},{mae_ours/mae_paper:.4f}\n')
    print(f'Saved to ./results/results.csv')


if __name__ == '__main__':
    main()

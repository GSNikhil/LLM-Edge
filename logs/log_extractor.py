import re

def parse_log(file_path):
    results = []
    with open(file_path, 'r') as file:
        data = file.read()
        blocks = data.split('==================================================')
        print(len(blocks))
        # print(blocks)
        for i, block in enumerate(blocks[1::2]):
            # print("=" * 100)
            print(i)
            print(block)
            if block == '\n' or block == '':
                continue

            if 'Pruning Layer idx' in block:
                # pruned_layers = int(re.search(r'Pruning Layer idx = (\d+)', block).group(1))
                accuracy = float(re.search(r'Model Accuracy on GSM8K: ([\d\.]+)%', block).group(1))
                rouge_1 = float(re.search(r'Average ROUGE-1: ([\d\.]+)', block).group(1))
                rouge_2 = float(re.search(r'Average ROUGE-2: ([\d\.]+)', block).group(1))
                rouge_l = float(re.search(r'Average ROUGE-L: ([\d\.]+)', block).group(1))
                perplexity = float(re.search(r'model perplexity: ([\d\.]+)', block).group(1)) if re.search(r'model perplexity: ([\d\.]+)', block) else "Inf"
                size = float(re.search(r'model size: ([\d\.]+) MiB', block).group(1))
                parameters = int(re.search(r'Model Parameters: (\d+)', block).group(1))
                
                results.append({
                    # 'Pruned idx': pruned_layers,
                    'Accuracy (%)': accuracy,
                    'ROUGE-1': rouge_1,
                    'ROUGE-2': rouge_2,
                    'ROUGE-L': rouge_l,
                    'Perplexity': perplexity,
                    'Size (MiB)': size,
                    'Parameters': parameters
                })
    
    return results

def main():
    file_path = 'D:/UCSD/WI25/ECE226 HW Acceleration of DNNs/LLM-Edge/logs/sensitivity_pruning.txt' # Path to your log file
    results = parse_log(file_path)
    print(results)
    for result in results:
        print(f"{result['Accuracy (%)']}\t{result['ROUGE-1']}\t{result['ROUGE-2']}\t{result['ROUGE-L']}\t{result['Perplexity']}\t{result['Size (MiB)']}\t{result['Parameters']}\t")

if __name__ == '__main__':
    main()

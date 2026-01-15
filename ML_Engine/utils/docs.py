
import ast
import os
import re

def parse_docstring(docstring):
    """Extract description, parameters, returns from docstring."""
    if not docstring:
        return {'description': '', 'returns': ''}
    
    # Simple parsing: split by sections
    lines = docstring.strip().split('\n')
    description = []
    returns = []
    current_section = 'description'
    
    for line in lines:
        line = line.strip()
        if line.lower().startswith('returns'):
            current_section = 'returns'
            continue
        if line.lower().startswith('parameters'):
            current_section = 'parameters'
            continue
        if line.lower().startswith('raises'):
            current_section = 'raises'
            continue
        
        if current_section == 'description':
            description.append(line)
        elif current_section == 'returns':
            returns.append(line)
    
    return {
        'description': ' '.join(description).strip(),
        'returns': ' '.join(returns).strip()
    }

def extract_functions(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            # Skip private functions (starting with _)
            if func_name.startswith('_'):
                continue
            # Get arguments
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            # Get docstring
            docstring = ast.get_docstring(node)
            parsed = parse_docstring(docstring)
            functions.append((func_name, args, parsed))
    return functions

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to the ML_Engine directory
    ml_engine_dir = os.path.abspath(os.path.join(script_dir, '..'))
    output_file = os.path.join(ml_engine_dir, 'FUNCTION_DOCUMENTATION.md')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('# ML Engine Function Documentation\n\n')
        f.write('## Overview\n')
        f.write('This document provides detailed information about all functions available in the ML_Engine package, including their parameters, return values, and purposes.\n\n')
        
        for root, dirs, files in os.walk(ml_engine_dir):
            # Skip the docs directory
            if 'docs' in dirs:
                dirs.remove('docs')
            
            for file in files:
                if file.endswith('.py') and file not in ['__init__.py', 'docs.py']:
                    filepath = os.path.join(root, file)
                    functions = extract_functions(filepath)
                    if not functions:
                        continue
                    
                    module_name = os.path.relpath(filepath, ml_engine_dir).replace(os.sep, '.')[:-3]
                    f.write(f'## {module_name}\n\n')
                    
                    for func_name, args, parsed in functions:
                        f.write(f'### {func_name}()\n')
                        f.write(f'**Signature**: {func_name}({", ".join(args)})\n\n')
                        if parsed['description']:
                            f.write(f'**Description**: {parsed["description"]}\n\n')
                        if parsed['returns']:
                            f.write(f'**Returns**: {parsed["returns"]}\n\n')
                        f.write('\n')
                    f.write('\n')
    
    print(f'Documentation generated: {output_file}')

if __name__ == '__main__':
    main()

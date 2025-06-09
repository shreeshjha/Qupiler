#!/usr/bin/env python3
"""
Example Usage Script for Enhanced Quantum Compilation Framework
Demonstrates how to use all components together
Fixed version with proper error handling
"""

import json
import os
import sys
from pathlib import Path

# Import the enhanced quantum compiler with error handling
try:
    from enhanced_quantum_compiler import (
        EnhancedQuantumCompiler, 
        QMLIRStandard, 
        ValidationLevel
    )
    COMPILER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Could not import enhanced quantum compiler: {e}")
    COMPILER_AVAILABLE = False

# Import optimization pipeline with error handling
try:
    from optimization_pipeline import OptimizationLevel
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    # Create a basic OptimizationLevel for compatibility
    class OptimizationLevel:
        NONE = "none"
        BASIC = "basic"  
        STANDARD = "standard"
        AGGRESSIVE = "aggressive"
    OPTIMIZATION_AVAILABLE = False

def create_example_ast():
    """Create an example AST JSON for quantum division"""
    
    example_ast = {
        "kind": "TranslationUnitDecl",
        "inner": [
            {
                "kind": "FunctionDecl",
                "name": "quantum_circuit",
                "attrs": [{"kind": "AnnotateAttr", "spelling": "__qpu__"}],
                "inner": [
                    {
                        "kind": "CompoundStmt",
                        "inner": [
                            {
                                "kind": "DeclStmt",
                                "inner": [
                                    {
                                        "kind": "VarDecl",
                                        "name": "dividend",
                                        "inner": [
                                            {"kind": "IntegerLiteral", "value": "15"}
                                        ]
                                    }
                                ]
                            },
                            {
                                "kind": "DeclStmt", 
                                "inner": [
                                    {
                                        "kind": "VarDecl",
                                        "name": "divisor",
                                        "inner": [
                                            {"kind": "IntegerLiteral", "value": "3"}
                                        ]
                                    }
                                ]
                            },
                            {
                                "kind": "DeclStmt",
                                "inner": [
                                    {
                                        "kind": "VarDecl",
                                        "name": "quotient",
                                        "inner": [
                                            {
                                                "kind": "BinaryOperator",
                                                "opcode": "/",
                                                "inner": [
                                                    {
                                                        "kind": "DeclRefExpr",
                                                        "referencedDecl": {"name": "dividend"}
                                                    },
                                                    {
                                                        "kind": "DeclRefExpr", 
                                                        "referencedDecl": {"name": "divisor"}
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            },
                            {
                                "kind": "CallExpr",
                                "inner": [
                                    {"name": "printf"},
                                    {"kind": "StringLiteral", "value": "Quotient: %d"},
                                    {
                                        "kind": "DeclRefExpr",
                                        "referencedDecl": {"name": "quotient"}
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    return example_ast

def run_basic_example():
    """Run basic compilation example"""
    print("🚀 Running Basic Compilation Example")
    print("="*50)
    
    if not COMPILER_AVAILABLE:
        print("❌ Enhanced quantum compiler not available, skipping...")
        return False
    
    # Create output directory
    output_dir = Path("example_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create example AST
    ast_data = create_example_ast()
    ast_path = output_dir / "quantum_division.json"
    
    with open(ast_path, 'w') as f:
        json.dump(ast_data, f, indent=2)
    
    print(f"📄 Created example AST: {ast_path}")
    
    # Create compiler with standard validation
    try:
        compiler = EnhancedQuantumCompiler(
            qmlir_standard=QMLIRStandard.QCOR,
            validation_level=ValidationLevel.STANDARD
        )
        print("✅ Compiler initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize compiler: {e}")
        return False
    
    # Define output paths
    qmlir_path = output_dir / "quantum_division.mlir"
    qiskit_path = output_dir / "quantum_division.py"
    qir_path = output_dir / "quantum_division.ll"
    qasm_path = output_dir / "quantum_division.qasm"
    
    # Run compilation
    try:
        success = compiler.compile_with_comprehensive_validation(
            ast_json_path=str(ast_path),
            output_qmlir_path=str(qmlir_path),
            output_qiskit_path=str(qiskit_path),
            output_qir_path=str(qir_path),
            output_qasm_path=str(qasm_path),
            target_hardware="ibm",
            enable_property_testing=True,
            optimization_level=OptimizationLevel.STANDARD
        )
        
        if success:
            print("\n✅ Basic compilation completed successfully!")
            print(f"📁 Output files in: {output_dir}")
            
            # Show generated files
            for filepath in [qmlir_path, qiskit_path, qir_path, qasm_path]:
                if filepath.exists():
                    print(f"   • {filepath.name}: {filepath.stat().st_size} bytes")
                else:
                    print(f"   • {filepath.name}: ❌ Not generated")
        else:
            print("\n❌ Basic compilation failed!")
            
        return success
        
    except Exception as e:
        print(f"\n💥 Compilation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_individual_components():
    """Demonstrate individual component usage"""
    print("\n🔬 Demonstrating Individual Components")
    print("="*50)
    
    # 1. Enhanced Validation Framework
    print("\n1️⃣  Enhanced Validation Framework:")
    try:
        from enhanced_dialect_module import EnhancedValidationFramework, ValidationLevel
        
        validation_framework = EnhancedValidationFramework(ValidationLevel.STRICT)
        
        # Example operations for validation
        test_operations = [
            "builtin.module {",
            "  quantum.func @quantum_circuit() attributes { qmlir.qpu_kernel = true } {",
            "    %q1 = quantum.alloc(4 : i32) : !quantum.Array<4>",
            "    %q2 = quantum.alloc(4 : i32) : !quantum.Array<4>",
            "    quantum.x %q1[0] : !quantum.Qubit",
            "    func.return",
            "  }",
            "}"
        ]
        
        # Test QCOR compliance
        compliance_result = validation_framework.validate_qcor_compliance(test_operations)
        print(f"   ✅ Validation Framework Available")
        print(f"   QCOR Compliance: {'PASS' if compliance_result.is_valid else 'FAIL'}")
        print(f"   Confidence Score: {compliance_result.confidence_score:.2%}")
        
    except ImportError as e:
        print(f"   ❌ Could not import validation framework: {e}")
    except Exception as e:
        print(f"   ⚠️  Validation framework error: {e}")
    
    # 2. Property-Based Testing
    print("\n2️⃣  Property-Based Testing Framework:")
    try:
        from property_testing_framework import MorphQFramework
        
        morphq = MorphQFramework(random_seed=42)
        
        test_circuit = [
            "quantum.x %q1 : !quantum.Qubit",
            "quantum.h %q2 : !quantum.Qubit", 
            "quantum.cnot %q1, %q2 : !quantum.Qubit, !quantum.Qubit"
        ]
        
        # Generate small test suite
        test_suite = morphq.generate_comprehensive_test_suite(
            test_circuit, num_mutation_tests=10, num_metamorphic_tests=5
        )
        
        print(f"   ✅ Property Testing Framework Available")
        print(f"   Generated {len(test_suite)} test cases")
        
    except ImportError as e:
        print(f"   ❌ Could not import property testing framework: {e}")
    except Exception as e:
        print(f"   ⚠️  Property testing framework error: {e}")
    
    # 3. Optimization Pipeline
    print("\n3️⃣  Optimization Pipeline:")
    try:
        from optimization_pipeline import OptimizationPipelineManager, OptimizationLevel
        
        optimizer = OptimizationPipelineManager()
        
        test_circuit = [
            "quantum.x %q1 : !quantum.Qubit",
            "quantum.x %q1 : !quantum.Qubit",  # Should be optimized away
            "quantum.h %q2 : !quantum.Qubit"
        ]
        
        # Run optimization
        optimized_circuit, metrics = optimizer.run_optimization_pipeline(
            test_circuit, ['standard']
        )
        
        print(f"   ✅ Optimization Pipeline Available")
        print(f"   Original gates: {len(test_circuit)}")
        print(f"   Optimized gates: {len(optimized_circuit)}")
        
    except ImportError as e:
        print(f"   ❌ Could not import optimization pipeline: {e}")
    except Exception as e:
        print(f"   ⚠️  Optimization pipeline error: {e}")
    
    # 4. Hardware Integration
    print("\n4️⃣  Hardware Integration:")
    try:
        from hardware_backend_integration import HardwareBackendFactory, HardwarePlatform
        
        # Try to create IBM backend
        try:
            ibm_backend = HardwareBackendFactory.create_backend(HardwarePlatform.IBM_QUANTUM)
            print(f"   ✅ Hardware Integration Available")
            
            # Test hardware validation
            test_circuit = ["quantum.x %q1 : !quantum.Qubit"]
            is_valid, errors = ibm_backend.validate_circuit(test_circuit)
            print(f"   Hardware validation: {'PASS' if is_valid else 'FAIL'}")
            
        except Exception as e:
            print(f"   ⚠️  Hardware backend error: {e}")
        
    except ImportError as e:
        print(f"   ❌ Could not import hardware integration: {e}")
    
    # 5. Direct dialect usage
    print("\n5️⃣  Direct Dialect Module Usage:")
    try:
        from enhanced_dialect_module import (
            EnhancedQuantumResourceManager,
            EnhancedValidationFramework,
            ValidationLevel,
            QubitType
        )
        
        # Create validation framework
        validator = EnhancedValidationFramework(ValidationLevel.STANDARD)
        
        # Create resource manager
        resource_manager = EnhancedQuantumResourceManager(validator)
        
        # Allocate some qubits
        allocation, result = resource_manager.allocate_qubits_with_validation(
            "test_qubits", 4, QubitType.DATA
        )
        
        print(f"   ✅ Dialect Module Available")
        print(f"   Allocated {allocation.bit_width} qubits")
        print(f"   Allocation valid: {result.is_valid}")
        
        # Get resource summary
        summary = resource_manager.get_resource_utilization_report()
        print(f"   Total qubits: {summary['total_qubits']}")
        print(f"   Efficiency: {summary['utilization_efficiency']:.2%}")
        
    except ImportError as e:
        print(f"   ❌ Could not import dialect module: {e}")
    except Exception as e:
        print(f"   ⚠️  Dialect module error: {e}")

def check_dependencies():
    """Check which dependencies are available"""
    print("\n🔍 Checking Dependencies")
    print("="*50)
    
    dependencies = {
        "Enhanced Dialect Module": "enhanced_dialect_module",
        "Property Testing Framework": "property_testing_framework", 
        "Optimization Pipeline": "optimization_pipeline",
        "Hardware Integration": "hardware_backend_integration"
    }
    
    available_modules = []
    
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}: Available")
            available_modules.append(module)
        except ImportError:
            print(f"❌ {name}: Not available")
    
    print(f"\n📊 {len(available_modules)}/{len(dependencies)} modules available")
    
    if len(available_modules) < len(dependencies):
        print("\n💡 Missing modules can be created by saving the provided code files:")
        for name, module in dependencies.items():
            if module not in [m.split('.')[-1] for m in available_modules]:
                print(f"   • Save {module}.py file")
    
    return available_modules

def main():
    """Main function to run all examples"""
    print("🌟 Quantum Compilation Framework - Example Usage")
    print("="*60)
    
    # Check dependencies first
    available_modules = check_dependencies()
    
    # Only run examples if we have the basic requirements
    if 'enhanced_dialect_module' in [m.split('.')[-1] for m in available_modules]:
        try:
            # Demonstrate individual components first
            demonstrate_individual_components()
            
            # Run basic example if compiler is available
            if COMPILER_AVAILABLE:
                success = run_basic_example()
            else:
                print("\n⚠️  Skipping compilation examples (compiler not available)")
                success = False
            
            # Summary
            print("\n" + "="*60)
            print("📋 EXAMPLE SUMMARY") 
            print("="*60)
            
            if COMPILER_AVAILABLE:
                if success:
                    print("✅ Basic compilation example: SUCCESS")
                else:
                    print("❌ Basic compilation example: FAILED")
            else:
                print("⚠️  Compilation examples: SKIPPED (dependencies missing)")
            
            print("✅ Component demonstration: COMPLETED")
            
            print("\n🎉 Example usage demonstration completed!")
            print("\n💡 Next steps:")
            print("   1. Install missing dependencies if needed")
            print("   2. Save all provided code files in the same directory") 
            print("   3. Modify the AST JSON to test your own quantum circuits")
            print("   4. Experiment with different validation levels")
            
        except Exception as e:
            print(f"\n💥 Example failed with error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n❌ Essential dialect module not available!")
        print("💡 Please save the enhanced_dialect_module.py file first")

if __name__ == "__main__":
    main()

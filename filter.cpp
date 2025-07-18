#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <algorithm>

// Compile: g++ -O3 -march=native -std=c++17 filter.cpp -o filter

struct Sequence {
    std::string header;
    std::string original_seq;
    std::string cleaned_seq;
    float coverage = 0.0f;
    float identity = 0.0f;
    int non_gap_count = 0;
    int aa_counts[26];
};

// Remove lowercase letters
std::string clean_sequence(const std::string& seq) {
    std::string result;
    result.reserve(seq.length());
    for (char c : seq) {
        if (!islower(c)) {
            result.push_back(c);
        }
    }
    return result;
}

// Calculate coverage and identity against query
void calculate_metrics(Sequence& seq, const std::string& query) {
    if (seq.cleaned_seq.length() != query.length()) {
        seq.coverage = 0.0f;
        seq.identity = 0.0f;
        return;
    }
    
    int gaps = 0;
    int matches = 0;
    const size_t query_len = query.length();
    
    for (size_t j = 0; j < query_len; j++) {
        if (seq.cleaned_seq[j] == '-') {
            gaps++;
        } else if (seq.cleaned_seq[j] == query[j]) {
            matches++;
        }
    }
    
    seq.coverage = 1.0f - (float)gaps / query_len;
    seq.identity = (float)matches / query_len;
}

// Analyze sequence composition
void analyze_sequence(Sequence& seq) {
    seq.non_gap_count = 0;
    memset(seq.aa_counts, 0, sizeof(seq.aa_counts));
    
    for (char c : seq.cleaned_seq) {
        if (c != '-') {
            seq.non_gap_count++;
            if (c >= 'A' && c <= 'Z') {
                seq.aa_counts[c - 'A']++;
            }
        }
    }
}

// Fast pairwise identity calculation
float calculate_pairwise_identity(const std::string& seq1, const std::string& seq2) {
    if (seq1.length() != seq2.length()) return 0.0f;
    
    int matches = 0;
    int valid = 0;
    
    for (size_t i = 0; i < seq1.length(); i++) {
        if (seq1[i] != '-' && seq2[i] != '-') {
            valid++;
            if (seq1[i] == seq2[i]) {
                matches++;
            }
        }
    }
    
    return valid > 0 ? (float)matches / valid : 0.0f;
}

// Quick pre-filter for redundancy check
bool should_skip_redundancy_check(const Sequence& s1, const Sequence& s2, float threshold) {
    // Length filter
    int diff = std::abs(s1.non_gap_count - s2.non_gap_count);
    if (diff > s1.non_gap_count * (1.0f - threshold)) {
        return true;
    }
    
    // Amino acid composition filter
    int aa_diff = 0;
    for (int i = 0; i < 26; i++) {
        aa_diff += std::abs(s1.aa_counts[i] - s2.aa_counts[i]);
    }
    
    if (aa_diff > s1.non_gap_count * (1.0f - threshold) * 1.5f) {
        return true;
    }
    
    return false;
}

void print_usage(const char* program) {
    std::cout << "Enhanced MSA Filter\n\n";
    std::cout << "Usage: " << program << " -i <input.a3m> -o <output.a3m> [options]\n";
    std::cout << "   or: " << program << " <input.a3m> <output.a3m> [options]\n\n";
    std::cout << "Required:\n";
    std::cout << "  -i <file>    Input a3m file\n";
    std::cout << "  -o <file>    Output filtered a3m file\n\n";
    std::cout << "Options:\n";
    std::cout << "  -id <int>    Maximum pairwise identity % (default: off)\n";
    std::cout << "  -qid <int>   Minimum query identity % (default: off)\n";
    std::cout << "  -cov <int>   Minimum coverage % (default: off)\n";
    std::cout << "  -h           Show this help\n\n";
    std::cout << "Note: By default, NO filtering is applied.\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program << " -i input.a3m -o output.a3m\n";
    std::cout << "  " << program << " -i input.a3m -o output.a3m -cov 75\n";
    std::cout << "  " << program << " input.a3m output.a3m -id 90 -cov 75 -qid 15\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string input_file;
    std::string output_file;
    float max_pairwise_identity = 0.0f;  // Default: off
    float min_query_identity = 0.0f;     // Default: off  
    float min_coverage = 0.0f;           // Default: off
    
    // Simple argument parsing
    int i = 1;
    while (i < argc) {
        std::string arg = argv[i];
        
        if (arg == "-i" && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-id" && i + 1 < argc) {
            max_pairwise_identity = std::stoi(argv[++i]) / 100.0f;
        } else if (arg == "-qid" && i + 1 < argc) {
            min_query_identity = std::stoi(argv[++i]) / 100.0f;
        } else if (arg == "-cov" && i + 1 < argc) {
            min_coverage = std::stoi(argv[++i]) / 100.0f;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] != '-' && input_file.empty()) {
            // Positional argument - input file
            input_file = arg;
        } else if (arg[0] != '-' && output_file.empty()) {
            // Positional argument - output file
            output_file = arg;
        }
        i++;
    }
    
    // Validate required arguments
    if (input_file.empty() || output_file.empty()) {
        std::cerr << "Error: Input and output files are required\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Show filter settings
    std::cout << "Filter Settings:\n";
    std::cout << "  Input: " << input_file << "\n";
    std::cout << "  Output: " << output_file << "\n";
    
    bool has_filters = false;
    if (max_pairwise_identity > 0.0f) {
        std::cout << "  Max pairwise identity: " << (int)(max_pairwise_identity * 100) << "%\n";
        has_filters = true;
    }
    if (min_query_identity > 0.0f) {
        std::cout << "  Min query identity: " << (int)(min_query_identity * 100) << "%\n";
        has_filters = true;
    }
    if (min_coverage > 0.0f) {
        std::cout << "  Min coverage: " << (int)(min_coverage * 100) << "%\n";
        has_filters = true;
    }
    if (!has_filters) {
        std::cout << "  No filtering applied\n";
    }
    std::cout << "\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Read sequences
    std::cout << "Reading sequences...\n";
    std::vector<Sequence> sequences;
    
    std::ifstream file(input_file);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << input_file << std::endl;
        return 1;
    }
    
    std::string line;
    std::string current_header;
    std::string current_seq;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        if (line[0] == '>') {
            if (!current_header.empty()) {
                Sequence seq;
                seq.header = current_header;
                seq.original_seq = current_seq;
                seq.cleaned_seq = clean_sequence(current_seq);
                sequences.push_back(seq);
            }
            current_header = line;
            current_seq.clear();
        } else {
            current_seq += line;
        }
    }
    
    if (!current_header.empty()) {
        Sequence seq;
        seq.header = current_header;
        seq.original_seq = current_seq;
        seq.cleaned_seq = clean_sequence(current_seq);
        sequences.push_back(seq);
    }
    
    file.close();
    
    std::cout << "Read " << sequences.size() << " sequences\n";
    
    if (sequences.empty()) {
        std::cerr << "Error: No sequences found\n";
        return 1;
    }
    
    // Get query sequence
    const std::string& query = sequences[0].cleaned_seq;
    const size_t query_len = query.length();
    
    // Calculate metrics for all sequences
    for (auto& seq : sequences) {
        calculate_metrics(seq, query);
        analyze_sequence(seq);
    }
    
    // Apply filters
    std::vector<Sequence> filtered_sequences;
    
    size_t length_filtered = 0;
    size_t coverage_filtered = 0;
    size_t qid_filtered = 0;
    
    for (size_t i = 0; i < sequences.size(); i++) {
        const auto& seq = sequences[i];
        
        // Always keep query
        if (i == 0) {
            filtered_sequences.push_back(seq);
            continue;
        }
        
        // Length filter
        if (seq.cleaned_seq.length() != query_len) {
            length_filtered++;
            continue;
        }
        
        // Coverage filter
        if (min_coverage > 0.0f && seq.coverage < min_coverage) {
            coverage_filtered++;
            continue;
        }
        
        // Query identity filter
        if (min_query_identity > 0.0f && seq.identity < min_query_identity) {
            qid_filtered++;
            continue;
        }
        
        filtered_sequences.push_back(seq);
    }
    
    std::cout << "After cov/qid filtering: " << filtered_sequences.size() << " sequences\n";
    if (length_filtered > 0) std::cout << "  Length filtered: " << length_filtered << "\n";
    if (coverage_filtered > 0) std::cout << "  Coverage filtered: " << coverage_filtered << "\n";
    if (qid_filtered > 0) std::cout << "  Query identity filtered: " << qid_filtered << "\n";
    
    // Apply redundancy filter if requested
    std::vector<Sequence> final_sequences;
    
    if (max_pairwise_identity > 0.0f) {
        std::cout << "Applying redundancy filter...\n";
        
        std::vector<bool> keep(filtered_sequences.size(), false);
        keep[0] = true;  // Always keep query
        
        std::vector<size_t> kept_indices = {0};
        size_t comparisons = 0;
        size_t skipped = 0;
        
        for (size_t i = 1; i < filtered_sequences.size(); i++) {
            if (i % 1000 == 0) {
                std::cout << "\rProcessing " << i << "/" << filtered_sequences.size() 
                         << " (kept: " << kept_indices.size() << ")";
                std::cout.flush();
            }
            
            bool should_keep = true;
            
            for (size_t kept_idx : kept_indices) {
                if (should_skip_redundancy_check(filtered_sequences[i], filtered_sequences[kept_idx], max_pairwise_identity)) {
                    skipped++;
                    continue;
                }
                
                comparisons++;
                
                float pairwise_identity = calculate_pairwise_identity(
                    filtered_sequences[i].cleaned_seq,
                    filtered_sequences[kept_idx].cleaned_seq
                );
                
                if (pairwise_identity > max_pairwise_identity) {
                    should_keep = false;
                    break;
                }
            }
            
            if (should_keep) {
                keep[i] = true;
                kept_indices.push_back(i);
            }
        }
        
        std::cout << "\n";
        
        for (size_t i = 0; i < filtered_sequences.size(); i++) {
            if (keep[i]) {
                final_sequences.push_back(std::move(filtered_sequences[i]));
            }
        }
        
        std::cout << "Redundancy filter stats:\n";
        std::cout << "  Comparisons: " << comparisons << "\n";
        std::cout << "  Skipped: " << skipped << "\n";
    } else {
        final_sequences = std::move(filtered_sequences);
    }
    
    // Write output
    std::cout << "Writing output...\n";
    
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot write to " << output_file << std::endl;
        return 1;
    }
    
    for (const auto& seq : final_sequences) {
        out << seq.header << '\n';
        out << seq.original_seq << '\n';
    }
    
    out.close();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    // Final statistics
    std::cout << "\nFinal Results:\n";
    std::cout << "  Input sequences: " << sequences.size() << "\n";
    std::cout << "  Output sequences: " << final_sequences.size() << "\n";
    std::cout << "  Kept: " << (100.0 * final_sequences.size() / sequences.size()) << "%\n";
    std::cout << "  Processing time: " << duration << " seconds\n";
    
    return 0;
}

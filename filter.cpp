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

// Fast pairwise identity calculation for redundancy filtering
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
    std::cout << "  -id <int>    Maximum pairwise identity % for redundancy filtering\n";
    std::cout << "               (only applied if specified)\n";
    std::cout << "  -qid <int>   Minimum query identity % (default: 0 - no filtering)\n";
    std::cout << "  -cov <int>   Minimum coverage % (default: 0 - no filtering)\n";
    std::cout << "  -h           Show this help\n\n";
    std::cout << "Note: By default, NO filtering is applied. Specify thresholds to enable filtering.\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program << " -i input.a3m -o output.a3m              # No filtering (copy all)\n";
    std::cout << "  " << program << " -i input.a3m -o output.a3m -cov 75      # Coverage filtering only\n";
    std::cout << "  " << program << " -i input.a3m -o output.a3m -id 90       # Redundancy filtering only\n";
    std::cout << "  " << program << " -i input.a3m -o output.a3m -cov 75 -qid 15 -id 90  # All filters\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    float max_pairwise_identity = 0.9f;
    float min_query_identity = 0.15f;
    float min_coverage = 0.75f;
    
    bool do_redundancy_filter = false;
    
    // Parse arguments
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-id" && i + 1 < argc) {
            max_pairwise_identity = std::stoi(argv[++i]) / 100.0f;
            do_redundancy_filter = true;
        } else if (arg == "-qid" && i + 1 < argc) {
            min_query_identity = std::stoi(argv[++i]) / 100.0f;
        } else if (arg == "-cov" && i + 1 < argc) {
            min_coverage = std::stoi(argv[++i]) / 100.0f;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    std::cout << "Enhanced Filter Parameters:" << std::endl;
    if (do_redundancy_filter) {
        std::cout << "  Max pairwise identity: " << (int)(max_pairwise_identity * 100) << "%" << std::endl;
    } else {
        std::cout << "  Redundancy filtering: DISABLED" << std::endl;
    }
    if (min_query_identity > 0.0f) {
        std::cout << "  Min query identity: " << (int)(min_query_identity * 100) << "%" << std::endl;
    } else {
        std::cout << "  Query identity filtering: DISABLED" << std::endl;
    }
    if (min_coverage > 0.0f) {
        std::cout << "  Min coverage: " << (int)(min_coverage * 100) << "%" << std::endl;
    } else {
        std::cout << "  Coverage filtering: DISABLED" << std::endl;
    }
    
    // Read sequences
    std::cout << "\nReading sequences..." << std::endl;
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
            // Save previous sequence
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
    
    // Don't forget last sequence
    if (!current_header.empty()) {
        Sequence seq;
        seq.header = current_header;
        seq.original_seq = current_seq;
        seq.cleaned_seq = clean_sequence(current_seq);
        sequences.push_back(seq);
    }
    
    file.close();
    
    std::cout << "Read " << sequences.size() << " sequences" << std::endl;
    
    if (sequences.empty()) {
        std::cerr << "Error: No sequences found" << std::endl;
        return 1;
    }
    
    // Get query sequence (first sequence)
    const std::string& query = sequences[0].cleaned_seq;
    const size_t query_len = query.length();
    
    std::cout << "Query length: " << query_len << std::endl;
    
    // Calculate metrics for all sequences
    std::cout << "Calculating coverage and identity..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (auto& seq : sequences) {
        calculate_metrics(seq, query);
        analyze_sequence(seq);
    }
    
    // Apply coverage and query identity filters
    std::cout << "Applying coverage and query identity filters..." << std::endl;
    
    std::vector<Sequence> filtered_sequences;
    filtered_sequences.reserve(sequences.size());
    
    // Always keep query (first sequence)
    filtered_sequences.push_back(std::move(sequences[0]));
    
    size_t coverage_filtered = 0;
    size_t qid_filtered = 0;
    size_t length_filtered = 0;
    
    for (size_t i = 1; i < sequences.size(); i++) {
        const auto& seq = sequences[i];
        
        // Length filter
        if (seq.cleaned_seq.length() != query_len) {
            length_filtered++;
            continue;
        }
        
        // Coverage filter
        if (seq.coverage < min_coverage) {
            coverage_filtered++;
            continue;
        }
        
        // Query identity filter
        if (seq.identity < min_query_identity) {
            qid_filtered++;
            continue;
        }
        
        filtered_sequences.push_back(sequences[i]);
    }
    
    std::cout << "After initial filtering:" << std::endl;
    std::cout << "  Length filtered: " << length_filtered << std::endl;
    std::cout << "  Coverage filtered: " << coverage_filtered << std::endl;
    std::cout << "  Query identity filtered: " << qid_filtered << std::endl;
    std::cout << "  Remaining: " << filtered_sequences.size() << std::endl;
    
    // Apply redundancy filter (pairwise identity) - only if requested
    std::vector<Sequence> final_sequences;
    size_t comparisons = 0;
    size_t skipped = 0;
    
    if (do_redundancy_filter) {
        std::cout << "Applying redundancy filter..." << std::endl;
        
        std::vector<bool> keep(filtered_sequences.size(), false);
        keep[0] = true;  // Always keep query
        
        std::vector<size_t> kept_indices;
        kept_indices.push_back(0);
        
        for (size_t i = 1; i < filtered_sequences.size(); i++) {
            if (i % 1000 == 0) {
                std::cout << "\rProcessing " << i << "/" << filtered_sequences.size() 
                         << " (kept: " << kept_indices.size() << ")";
                std::cout.flush();
            }
            
            bool should_keep = true;
            
            // Check against all kept sequences
            for (size_t kept_idx : kept_indices) {
                // Quick pre-filter
                if (should_skip_redundancy_check(filtered_sequences[i], filtered_sequences[kept_idx], max_pairwise_identity)) {
                    skipped++;
                    continue;
                }
                
                comparisons++;
                
                // Full pairwise identity check
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
        
        std::cout << std::endl;
        
        // Copy kept sequences
        for (size_t i = 0; i < filtered_sequences.size(); i++) {
            if (keep[i]) {
                final_sequences.push_back(std::move(filtered_sequences[i]));
            }
        }
    } else {
        std::cout << "Skipping redundancy filter (not requested)" << std::endl;
        final_sequences = std::move(filtered_sequences);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    // Write output
    std::cout << "Writing output..." << std::endl;
    
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
    
    // Report final statistics
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "  Input sequences: " << sequences.size() << std::endl;
    std::cout << "  After cov/qid filtering: " << filtered_sequences.size() << std::endl;
    std::cout << "  Final output sequences: " << final_sequences.size() << std::endl;
    std::cout << "  Overall kept: " << (100.0 * final_sequences.size() / sequences.size()) << "%" << std::endl;
    if (do_redundancy_filter) {
        std::cout << "  Pairwise comparisons: " << comparisons << std::endl;
        std::cout << "  Skipped comparisons: " << skipped << std::endl;
    }
    std::cout << "  Processing time: " << duration << " seconds" << std::endl;
    
    return 0;
}

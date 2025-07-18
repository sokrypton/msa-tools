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
    int non_gap_count;
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

// Analyze sequence
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

// Fast identity calculation
float calculate_identity(const std::string& seq1, const std::string& seq2) {
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

// Quick pre-filter
bool should_skip(const Sequence& s1, const Sequence& s2, float threshold) {
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

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.a3m> <output.a3m> [-id <value>]" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    float max_identity = 0.9f;
    
    // Parse arguments
    for (int i = 3; i < argc - 1; i++) {
        if (std::string(argv[i]) == "-id") {
            max_identity = std::stoi(argv[i + 1]) / 100.0f;
        }
    }
    
    std::cout << "filter - max identity: " << (int)(max_identity * 100) << "%" << std::endl;
    
    // Read sequences
    std::cout << "Reading sequences..." << std::endl;
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
                analyze_sequence(seq);
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
        analyze_sequence(seq);
        sequences.push_back(seq);
    }
    
    file.close();
    
    std::cout << "Read " << sequences.size() << " sequences" << std::endl;
    
    // Check if all have same length
    size_t expected_len = sequences[0].cleaned_seq.length();
    bool same_length = true;
    for (const auto& seq : sequences) {
        if (seq.cleaned_seq.length() != expected_len) {
            same_length = false;
            break;
        }
    }
    
    if (!same_length) {
        std::cout << "Warning: Sequences have different lengths after cleaning" << std::endl;
    }
    
    // Filter sequences
    std::cout << "Filtering sequences..." << std::endl;
    
    std::vector<bool> keep(sequences.size(), false);
    keep[0] = true;  // Always keep query
    
    std::vector<size_t> kept_indices;
    kept_indices.push_back(0);
    
    size_t comparisons = 0;
    size_t skipped = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 1; i < sequences.size(); i++) {
        if (i % 1000 == 0) {
            std::cout << "\rProcessing " << i << "/" << sequences.size() 
                     << " (kept: " << kept_indices.size() << ")";
            std::cout.flush();
        }
        
        // Skip if different length
        if (sequences[i].cleaned_seq.length() != expected_len) {
            continue;
        }
        
        bool should_keep = true;
        
        // Check against all kept sequences
        for (size_t kept_idx : kept_indices) {
            // Quick pre-filter
            if (should_skip(sequences[i], sequences[kept_idx], max_identity)) {
                skipped++;
                continue;
            }
            
            comparisons++;
            
            // Full identity check
            float identity = calculate_identity(
                sequences[i].cleaned_seq,
                sequences[kept_idx].cleaned_seq
            );
            
            if (identity > max_identity) {
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
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    // Write output
    std::cout << "Writing output..." << std::endl;
    
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot write to " << output_file << std::endl;
        return 1;
    }
    
    size_t written = 0;
    for (size_t i = 0; i < sequences.size(); i++) {
        if (keep[i]) {
            out << sequences[i].header << '\n';
            out << sequences[i].original_seq << '\n';
            written++;
        }
    }
    
    out.close();
    
    // Report statistics
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Input sequences: " << sequences.size() << std::endl;
    std::cout << "  Output sequences: " << written << std::endl;
    std::cout << "  Kept: " << (100.0 * written / sequences.size()) << "%" << std::endl;
    std::cout << "  Comparisons: " << comparisons << std::endl;
    std::cout << "  Skipped: " << skipped << std::endl;
    std::cout << "  Time: " << duration << " seconds" << std::endl;
    
    return 0;
}

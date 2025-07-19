#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD includes
#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2
#elif defined(__SSE2__)
#include <emmintrin.h>
#define USE_SSE2
#endif

// Compile with OpenMP and SIMD: g++ -O3 -march=native -std=c++17 -fopenmp filter.cpp -o filter
// Without OpenMP: g++ -O3 -march=native -std=c++17 filter.cpp -o filter

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

// Fast pairwise identity calculation with SIMD vectorization and early termination
float calculate_pairwise_identity(const std::string& seq1, const std::string& seq2, float threshold = 0.0f) {
    if (seq1.length() != seq2.length()) return 0.0f;
    
    const size_t len = seq1.length();
    const char* s1 = seq1.c_str();
    const char* s2 = seq2.c_str();
    
    int matches = 0;
    int valid = 0;
    
#ifdef USE_AVX2
    // AVX2 vectorized version - process 32 bytes at a time
    const size_t simd_end = len - (len % 32);
    
    __m256i matches_vec = _mm256_setzero_si256();
    __m256i valid_vec = _mm256_setzero_si256();
    __m256i gap_char = _mm256_set1_epi8('-');
    
    for (size_t i = 0; i < simd_end; i += 32) {
        // Load 32 characters from each sequence
        __m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
        __m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + i));
        
        // Check for gaps
        __m256i gap1 = _mm256_cmpeq_epi8(v1, gap_char);
        __m256i gap2 = _mm256_cmpeq_epi8(v2, gap_char);
        __m256i no_gaps = _mm256_andnot_si256(_mm256_or_si256(gap1, gap2), _mm256_set1_epi8(-1));
        
        // Check for matches
        __m256i matches_mask = _mm256_and_si256(_mm256_cmpeq_epi8(v1, v2), no_gaps);
        
        // Count valid positions and matches
        valid_vec = _mm256_sub_epi8(valid_vec, no_gaps);  // -1 for valid, 0 for invalid
        matches_vec = _mm256_sub_epi8(matches_vec, matches_mask);  // -1 for match, 0 for no match
    }
    
    // Horizontal sum of vectors
    __m256i valid_sum = _mm256_sad_epu8(valid_vec, _mm256_setzero_si256());
    __m256i matches_sum = _mm256_sad_epu8(matches_vec, _mm256_setzero_si256());
    
    // Extract results
    valid += _mm256_extract_epi64(valid_sum, 0) + _mm256_extract_epi64(valid_sum, 1) + 
             _mm256_extract_epi64(valid_sum, 2) + _mm256_extract_epi64(valid_sum, 3);
    matches += _mm256_extract_epi64(matches_sum, 0) + _mm256_extract_epi64(matches_sum, 1) + 
               _mm256_extract_epi64(matches_sum, 2) + _mm256_extract_epi64(matches_sum, 3);
    
    // Process remaining characters
    for (size_t i = simd_end; i < len; i++) {
        if (s1[i] != '-' && s2[i] != '-') {
            valid++;
            if (s1[i] == s2[i]) {
                matches++;
            }
        }
    }
    
#elif defined(USE_SSE2)
    // SSE2 vectorized version - process 16 bytes at a time
    const size_t simd_end = len - (len % 16);
    
    __m128i matches_vec = _mm_setzero_si128();
    __m128i valid_vec = _mm_setzero_si128();
    __m128i gap_char = _mm_set1_epi8('-');
    
    for (size_t i = 0; i < simd_end; i += 16) {
        // Load 16 characters from each sequence
        __m128i v1 = _mm_loadu_si128((__m128i*)(s1 + i));
        __m128i v2 = _mm_loadu_si128((__m128i*)(s2 + i));
        
        // Check for gaps
        __m128i gap1 = _mm_cmpeq_epi8(v1, gap_char);
        __m128i gap2 = _mm_cmpeq_epi8(v2, gap_char);
        __m128i no_gaps = _mm_andnot_si128(_mm_or_si128(gap1, gap2), _mm_set1_epi8(-1));
        
        // Check for matches
        __m128i matches_mask = _mm_and_si128(_mm_cmpeq_epi8(v1, v2), no_gaps);
        
        // Count valid positions and matches
        valid_vec = _mm_sub_epi8(valid_vec, no_gaps);
        matches_vec = _mm_sub_epi8(matches_vec, matches_mask);
    }
    
    // Horizontal sum
    __m128i valid_sum = _mm_sad_epu8(valid_vec, _mm_setzero_si128());
    __m128i matches_sum = _mm_sad_epu8(matches_vec, _mm_setzero_si128());
    
    valid += _mm_extract_epi16(valid_sum, 0) + _mm_extract_epi16(valid_sum, 4);
    matches += _mm_extract_epi16(matches_sum, 0) + _mm_extract_epi16(matches_sum, 4);
    
    // Process remaining characters
    for (size_t i = simd_end; i < len; i++) {
        if (s1[i] != '-' && s2[i] != '-') {
            valid++;
            if (s1[i] == s2[i]) {
                matches++;
            }
        }
    }
    
#else
    // Scalar fallback with early termination
    for (size_t i = 0; i < len; i++) {
        if (s1[i] != '-' && s2[i] != '-') {
            valid++;
            if (s1[i] == s2[i]) {
                matches++;
            }
            
            // Early termination check every 64 positions
            if (threshold > 0.0f && i > 64 && (i % 64) == 0) {
                float current_identity = (float)matches / valid;
                float max_possible = (float)(matches + (len - i)) / (valid + (len - i));
                if (max_possible < threshold) {
                    return current_identity;
                }
            }
        }
    }
#endif
    
    return valid > 0 ? (float)matches / valid : 0.0f;
}

// Traditional pre-filter (amino acid composition)
bool should_skip_composition_check(const Sequence& s1, const Sequence& s2, float threshold) {
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
    std::cout << "MSA Filter\n\n";
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
    std::cout << "  " << program << " -i input.a3m -o output.a3m -id 90\n";
    std::cout << "  " << program << " input.a3m output.a3m -id 90\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string input_file;
    std::string output_file;
    float max_pairwise_identity = 0.0f;
    float min_query_identity = 0.0f;
    float min_coverage = 0.0f;
    
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
            input_file = arg;
        } else if (arg[0] != '-' && output_file.empty()) {
            output_file = arg;
        }
        i++;
    }
    
    // Validate arguments
    if (input_file.empty() || output_file.empty()) {
        std::cerr << "Error: Input and output files are required\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Show filter settings
    std::cout << "Filter Settings:\n";
    std::cout << "  Input: " << input_file << "\n";
    std::cout << "  Output: " << output_file << "\n";
    
    // Show SIMD capabilities
    #ifdef USE_AVX2
    std::cout << "  SIMD: AVX2 enabled (32-byte vectorization)\n";
    #elif defined(USE_SSE2)
    std::cout << "  SIMD: SSE2 enabled (16-byte vectorization)\n";
    #else
    std::cout << "  SIMD: Scalar fallback (no vectorization)\n";
    #endif
    
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
    std::cout << "Calculating metrics...\n";
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < sequences.size(); i++) {
        auto& seq = sequences[i];
        calculate_metrics(seq, query);
        analyze_sequence(seq);
    }
    
    // Apply initial filters
    std::vector<Sequence> filtered_sequences;
    
    size_t length_filtered = 0;
    size_t coverage_filtered = 0;
    size_t qid_filtered = 0;
    
    for (size_t i = 0; i < sequences.size(); i++) {
        const auto& seq = sequences[i];
        
        // Always keep query
        if (i == 0) {
            filtered_sequences.push_back(std::move(sequences[i]));
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
        
        filtered_sequences.push_back(std::move(sequences[i]));
    }
    
    std::cout << "After cov/qid filtering: " << filtered_sequences.size() << " sequences\n";
    if (length_filtered > 0) std::cout << "  Length filtered: " << length_filtered << "\n";
    if (coverage_filtered > 0) std::cout << "  Coverage filtered: " << coverage_filtered << "\n";
    if (qid_filtered > 0) std::cout << "  Query identity filtered: " << qid_filtered << "\n";
    
    // Apply redundancy filter if requested
    std::vector<Sequence> final_sequences;
    
    if (max_pairwise_identity > 0.0f) {
        std::cout << "Applying redundancy filter...\n";
        
        #ifdef _OPENMP
        std::cout << "Using " << omp_get_max_threads() << " threads\n";
        #endif
        
        std::vector<bool> keep(filtered_sequences.size(), false);
        keep[0] = true;  // Always keep query
        
        std::vector<size_t> kept_indices = {0};
        size_t comparisons = 0;
        size_t skipped_composition = 0;
        
        // Process sequences in chunks
        const size_t chunk_size = 50;
        
        for (size_t start = 1; start < filtered_sequences.size(); start += chunk_size) {
            size_t end = std::min(start + chunk_size, filtered_sequences.size());
            
            if (start % 1000 == 1) {
                std::cout << "\rProcessing " << start << "/" << filtered_sequences.size() 
                         << " (kept: " << kept_indices.size() << ")";
                std::cout.flush();
            }
            
            // Thread-local counters
            std::vector<size_t> local_comparisons(omp_get_max_threads(), 0);
            std::vector<size_t> local_skipped_comp(omp_get_max_threads(), 0);
            std::vector<bool> chunk_keep(end - start, false);
            
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for (size_t idx = 0; idx < end - start; idx++) {
                size_t i = start + idx;
                bool should_keep = true;
                
                #ifdef _OPENMP
                int thread_id = omp_get_thread_num();
                #else
                int thread_id = 0;
                #endif
                
                // Check against all currently kept sequences
                for (size_t kept_idx : kept_indices) {
                    // Traditional composition pre-filter
                    if (should_skip_composition_check(filtered_sequences[i], filtered_sequences[kept_idx], max_pairwise_identity)) {
                        local_skipped_comp[thread_id]++;
                        continue;
                    }
                    
                    local_comparisons[thread_id]++;
                    
                    float pairwise_identity = calculate_pairwise_identity(
                        filtered_sequences[i].cleaned_seq,
                        filtered_sequences[kept_idx].cleaned_seq,
                        max_pairwise_identity  // For early termination
                    );
                    
                    if (pairwise_identity > max_pairwise_identity) {
                        should_keep = false;
                        break;
                    }
                }
                
                chunk_keep[idx] = should_keep;
            }
            
            // Update global state (serial section)
            for (size_t idx = 0; idx < end - start; idx++) {
                if (chunk_keep[idx]) {
                    size_t i = start + idx;
                    keep[i] = true;
                    kept_indices.push_back(i);
                }
            }
            
            // Accumulate thread-local counters
            for (size_t tc : local_comparisons) comparisons += tc;
            for (size_t ts : local_skipped_comp) skipped_composition += ts;
        }
        
        std::cout << "\n";
        
        for (size_t i = 0; i < filtered_sequences.size(); i++) {
            if (keep[i]) {
                final_sequences.push_back(std::move(filtered_sequences[i]));
            }
        }
        
        std::cout << "Redundancy filter stats:\n";
        std::cout << "  Full pairwise comparisons: " << comparisons << "\n";
        std::cout << "  Skipped (composition): " << skipped_composition << "\n";
        std::cout << "  Skip rate: " << (100.0 * skipped_composition / (comparisons + skipped_composition)) << "%\n";
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

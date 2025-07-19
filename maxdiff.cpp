#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <bitset>
#include <cmath>
#include <limits>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD includes
#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2
#define VECSIZE_INT 32
#elif defined(__SSE2__)
#include <emmintrin.h>
#define USE_SSE2
#define VECSIZE_INT 16
#else
#define VECSIZE_INT 1
#endif

// Compile with: g++ -O3 -march=native -std=c++17 -fopenmp maxdiff.cpp -o maxdiff

struct Sequence {
    std::string header;
    std::string original_seq;
    std::string cleaned_seq;  // Without lowercase letters
    float coverage = 0.0f;
    int non_gap_count = 0;
    int first_residue = -1;   // First non-gap position
    int last_residue = -1;    // Last non-gap position
    
    // Precomputed features for fast comparison
    std::vector<uint8_t> aa_sequence;  // Amino acid indices (0-19, 20=gap)
    int aa_counts[20] = {0};           // Amino acid composition
    
    // For fast SIMD comparison
    uint8_t* aligned_seq = nullptr;   // Aligned to SIMD boundaries
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

// Convert AA to index
inline int aa2idx(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c == '-') return 20;
    return 21; // Unknown
}

// Analyze sequence and compute features
void analyze_sequence(Sequence& seq, int alignment_length) {
    seq.non_gap_count = 0;
    memset(seq.aa_counts, 0, sizeof(seq.aa_counts));
    seq.aa_sequence.resize(seq.cleaned_seq.length());
    
    // Convert to indices and count
    for (size_t i = 0; i < seq.cleaned_seq.length(); i++) {
        int idx = aa2idx(seq.cleaned_seq[i]);
        seq.aa_sequence[i] = idx;
        
        if (idx < 20) {  // Not a gap
            seq.non_gap_count++;
            seq.aa_counts[idx]++;
            if (seq.first_residue == -1) seq.first_residue = i;
            seq.last_residue = i;
        }
    }
    
    seq.coverage = (float)seq.non_gap_count / seq.cleaned_seq.length();
    
    // Prepare SIMD-aligned sequence
    int simd_length = ((alignment_length + VECSIZE_INT - 1) / VECSIZE_INT) * VECSIZE_INT;
    seq.aligned_seq = (uint8_t*)aligned_alloc(VECSIZE_INT, simd_length);
    memset(seq.aligned_seq, 20, simd_length); // Fill with gaps
    memcpy(seq.aligned_seq, seq.aa_sequence.data(), seq.aa_sequence.size());
}

// Fast distance calculation with SIMD
float calculate_distance_simd(const Sequence& seq1, const Sequence& seq2) {
    if (seq1.non_gap_count == 0 || seq2.non_gap_count == 0) return 1.0f;
    
    // Quick pre-filter based on AA composition
    int comp_diff = 0;
    for (int i = 0; i < 20; i++) {
        comp_diff += std::abs(seq1.aa_counts[i] - seq2.aa_counts[i]);
    }
    float max_possible_diff = (float)comp_diff / (seq1.non_gap_count + seq2.non_gap_count);
    if (max_possible_diff < 0.1f) return 0.1f; // Very similar composition
    
    int first = std::max(seq1.first_residue, seq2.first_residue);
    int last = std::min(seq1.last_residue, seq2.last_residue);
    
    if (first > last) return 1.0f; // No overlap
    
    const uint8_t* s1 = seq1.aligned_seq;
    const uint8_t* s2 = seq2.aligned_seq;
    
    int matches = 0;
    int comparisons = 0;
    
#ifdef USE_AVX2
    // AVX2 version - process 32 positions at once
    const __m256i gap_vec = _mm256_set1_epi8(20);
    
    int i = (first / 32) * 32; // Start at aligned boundary
    int end = ((last + 1 + 31) / 32) * 32; // End at aligned boundary
    
    for (; i < end; i += 32) {
        __m256i v1 = _mm256_load_si256((__m256i*)(s1 + i));
        __m256i v2 = _mm256_load_si256((__m256i*)(s2 + i));
        
        // Find positions where neither sequence has a gap
        __m256i gap1 = _mm256_cmpeq_epi8(v1, gap_vec);
        __m256i gap2 = _mm256_cmpeq_epi8(v2, gap_vec);
        __m256i no_gaps = _mm256_andnot_si256(_mm256_or_si256(gap1, gap2), _mm256_set1_epi8(-1));
        
        // Find matches
        __m256i match = _mm256_cmpeq_epi8(v1, v2);
        __m256i valid_match = _mm256_and_si256(match, no_gaps);
        
        // Count
        uint32_t no_gaps_mask = _mm256_movemask_epi8(no_gaps);
        uint32_t match_mask = _mm256_movemask_epi8(valid_match);
        
        // Only count within our range
        for (int j = 0; j < 32 && i + j <= last; j++) {
            if (i + j >= first) {
                if (no_gaps_mask & (1 << j)) {
                    comparisons++;
                    if (match_mask & (1 << j)) {
                        matches++;
                    }
                }
            }
        }
    }
    
#elif defined(USE_SSE2)
    // SSE2 version - process 16 positions at once
    const __m128i gap_vec = _mm_set1_epi8(20);
    
    int i = (first / 16) * 16;
    int end = ((last + 1 + 15) / 16) * 16;
    
    for (; i < end; i += 16) {
        __m128i v1 = _mm_load_si128((__m128i*)(s1 + i));
        __m128i v2 = _mm_load_si128((__m128i*)(s2 + i));
        
        __m128i gap1 = _mm_cmpeq_epi8(v1, gap_vec);
        __m128i gap2 = _mm_cmpeq_epi8(v2, gap_vec);
        __m128i no_gaps = _mm_andnot_si128(_mm_or_si128(gap1, gap2), _mm_set1_epi8(-1));
        
        __m128i match = _mm_cmpeq_epi8(v1, v2);
        __m128i valid_match = _mm_and_si128(match, no_gaps);
        
        uint16_t no_gaps_mask = _mm_movemask_epi8(no_gaps);
        uint16_t match_mask = _mm_movemask_epi8(valid_match);
        
        for (int j = 0; j < 16 && i + j <= last; j++) {
            if (i + j >= first) {
                if (no_gaps_mask & (1 << j)) {
                    comparisons++;
                    if (match_mask & (1 << j)) {
                        matches++;
                    }
                }
            }
        }
    }
    
#else
    // Scalar fallback
    for (int i = first; i <= last; i++) {
        if (s1[i] < 20 && s2[i] < 20) {
            comparisons++;
            if (s1[i] == s2[i]) {
                matches++;
            }
        }
    }
#endif
    
    if (comparisons < 10) return 1.0f; // Too few comparisons
    
    return 1.0f - (float)matches / comparisons;
}

// Smart sequence selection with diversity maximization
std::vector<int> select_diverse_sequences(std::vector<Sequence>& sequences, int n_select, bool verbose) {
    std::vector<int> selected;
    std::vector<bool> is_selected(sequences.size(), false);
    std::vector<float> min_distances(sequences.size(), std::numeric_limits<float>::max());
    
    // Position coverage tracking
    int L = sequences[0].cleaned_seq.length();
    std::vector<int> position_coverage(L, 0);
    
    // Always include query
    selected.push_back(0);
    is_selected[0] = true;
    
    // Update position coverage for query
    for (int i = 0; i < L; i++) {
        if (sequences[0].aa_sequence[i] < 20) {
            position_coverage[i]++;
        }
    }
    
    // Update min distances from query
    #pragma omp parallel for
    for (int k = 1; k < sequences.size(); k++) {
        min_distances[k] = calculate_distance_simd(sequences[0], sequences[k]);
    }
    
    // Main selection loop
    while (selected.size() < n_select) {
        float best_score = -1.0f;
        int best_idx = -1;
        
        // Find sequence with maximum minimum distance (MaxMin criterion)
        #pragma omp parallel
        {
            float local_best_score = -1.0f;
            int local_best_idx = -1;
            
            #pragma omp for nowait
            for (int k = 1; k < sequences.size(); k++) {
                if (!is_selected[k] && sequences[k].coverage >= 0.3f) {
                    // Score combines minimum distance and position coverage bonus
                    float score = min_distances[k];
                    
                    // Add small bonus for covering rare positions
                    float position_bonus = 0;
                    int rare_positions = 0;
                    for (int i = sequences[k].first_residue; i <= sequences[k].last_residue; i++) {
                        if (sequences[k].aa_sequence[i] < 20 && position_coverage[i] < 5) {
                            rare_positions++;
                        }
                    }
                    position_bonus = 0.1f * rare_positions / sequences[k].non_gap_count;
                    
                    score = score * 0.9f + position_bonus * 0.1f;
                    
                    if (score > local_best_score) {
                        local_best_score = score;
                        local_best_idx = k;
                    }
                }
            }
            
            #pragma omp critical
            {
                if (local_best_score > best_score) {
                    best_score = local_best_score;
                    best_idx = local_best_idx;
                }
            }
        }
        
        if (best_idx < 0) break;
        
        selected.push_back(best_idx);
        is_selected[best_idx] = true;
        
        // Update position coverage
        for (int i = 0; i < L; i++) {
            if (sequences[best_idx].aa_sequence[i] < 20) {
                position_coverage[i]++;
            }
        }
        
        // Update minimum distances for remaining sequences
        #pragma omp parallel for
        for (int k = 1; k < sequences.size(); k++) {
            if (!is_selected[k]) {
                float dist = calculate_distance_simd(sequences[best_idx], sequences[k]);
                min_distances[k] = std::min(min_distances[k], dist);
            }
        }
        
        // Progress report
        if (selected.size() % 10 == 0 || verbose) {
            std::cout << "\rSelected " << selected.size() << "/" << n_select 
                      << " (min dist: " << best_score << ")";
            std::cout.flush();
        }
    }
    
    std::cout << "\n";
    return selected;
}

void print_usage(const char* program) {
    std::cout << "MaxDiff - Maximum Diversity Selection for MSA\n\n";
    std::cout << "Usage: " << program << " -i <input.a3m> -o <output.a3m> -diff <N> [options]\n\n";
    std::cout << "Required:\n";
    std::cout << "  -i <file>    Input a3m file\n";
    std::cout << "  -o <file>    Output filtered a3m file\n";
    std::cout << "  -diff <int>  Number of maximally diverse sequences to select\n\n";
    std::cout << "Options:\n";
    std::cout << "  -cov <int>   Minimum coverage % (default: 30)\n";
    std::cout << "  -v           Verbose output\n";
    std::cout << "  -h           Show this help\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string input_file;
    std::string output_file;
    int n_select = 100;
    float min_coverage = 0.3f;
    bool verbose = false;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-i" && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-diff" && i + 1 < argc) {
            n_select = std::stoi(argv[++i]);
        } else if (arg == "-cov" && i + 1 < argc) {
            min_coverage = std::stoi(argv[++i]) / 100.0f;
        } else if (arg == "-v") {
            verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (input_file.empty() || output_file.empty()) {
        std::cerr << "Error: Input and output files are required\n";
        print_usage(argv[0]);
        return 1;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Read sequences
    std::cout << "Reading sequences from " << input_file << "...\n";
    std::vector<Sequence> sequences;
    
    std::ifstream file(input_file);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << input_file << std::endl;
        return 1;
    }
    
    std::string line;
    Sequence current_seq;
    bool in_sequence = false;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        if (line[0] == '>') {
            if (in_sequence && !current_seq.original_seq.empty()) {
                current_seq.cleaned_seq = clean_sequence(current_seq.original_seq);
                sequences.push_back(std::move(current_seq));
                current_seq = Sequence();
            }
            current_seq.header = line;
            in_sequence = true;
        } else if (in_sequence) {
            current_seq.original_seq += line;
        }
    }
    
    if (in_sequence && !current_seq.original_seq.empty()) {
        current_seq.cleaned_seq = clean_sequence(current_seq.original_seq);
        sequences.push_back(std::move(current_seq));
    }
    
    file.close();
    
    std::cout << "Read " << sequences.size() << " sequences\n";
    
    if (sequences.empty()) {
        std::cerr << "Error: No sequences found in input file\n";
        return 1;
    }
    
    int alignment_length = sequences[0].cleaned_seq.length();
    std::cout << "Alignment length: " << alignment_length << "\n";
    
    // Analyze all sequences
    std::cout << "Analyzing sequences...\n";
    #pragma omp parallel for
    for (int i = 0; i < sequences.size(); i++) {
        analyze_sequence(sequences[i], alignment_length);
    }
    
    // Count sequences passing coverage filter
    int n_valid = 0;
    for (const auto& seq : sequences) {
        if (seq.coverage >= min_coverage) n_valid++;
    }
    std::cout << "Sequences with coverage >= " << (min_coverage * 100) << "%: " << n_valid << "\n";
    
    #ifdef _OPENMP
    std::cout << "Using " << omp_get_max_threads() << " threads\n";
    #endif
    
    // Perform diversity selection
    std::cout << "\nSelecting " << n_select << " diverse sequences...\n";
    auto selected = select_diverse_sequences(sequences, n_select, verbose);
    
    // Calculate statistics
    float avg_coverage = 0.0f;
    float avg_pairwise_dist = 0.0f;
    int n_pairs = 0;
    
    for (int idx : selected) {
        avg_coverage += sequences[idx].coverage;
    }
    avg_coverage /= selected.size();
    
    // Sample pairwise distances
    std::cout << "Calculating statistics...\n";
    for (size_t i = 0; i < std::min((size_t)20, selected.size()); i++) {
        for (size_t j = i + 1; j < std::min((size_t)20, selected.size()); j++) {
            avg_pairwise_dist += calculate_distance_simd(sequences[selected[i]], sequences[selected[j]]);
            n_pairs++;
        }
    }
    if (n_pairs > 0) avg_pairwise_dist /= n_pairs;
    
    // Write output
    std::cout << "Writing " << selected.size() << " sequences to " << output_file << "...\n";
    
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot write to " << output_file << std::endl;
        return 1;
    }
    
    for (int idx : selected) {
        out << sequences[idx].header << '\n';
        out << sequences[idx].original_seq << '\n';
    }
    
    out.close();
    
    // Clean up aligned sequences
    for (auto& seq : sequences) {
        if (seq.aligned_seq) {
            free(seq.aligned_seq);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // Final statistics
    std::cout << "\nResults:\n";
    std::cout << "  Selected sequences: " << selected.size() << "\n";
    std::cout << "  Average coverage: " << (avg_coverage * 100) << "%\n";
    std::cout << "  Average pairwise distance: " << (avg_pairwise_dist * 100) << "%\n";
    std::cout << "  Processing time: " << (duration / 1000.0) << " seconds\n";
    
    if (verbose) {
        std::cout << "\nFirst 10 selected sequences:\n";
        for (size_t i = 0; i < std::min((size_t)10, selected.size()); i++) {
            const auto& seq = sequences[selected[i]];
            std::cout << "  " << i+1 << ". " << seq.header 
                      << " (cov: " << (seq.coverage * 100) << "%)\n";
        }
    }
    
    return 0;
}

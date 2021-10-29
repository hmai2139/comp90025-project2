
/*
*  COMP90025 S2 2021, Project 2.
*  CPP program to solve the sequence alignment problem.
*  Adapted from https://www.geeksforgeeks.org/sequence-alignment-problem/.
*  Parallelisation using OpenMPI and OpenMP.
*  Reused OpenMP codes from Project 1.
*  Hoang Viet Mai, 813361 <vietm@student.unimelb.edu.au>.
*/

#include <mpi.h>
#include <sys/time.h>
#include <string>
#include <cstring>
#include <iostream>
#include "sha512.hh"
#include <omp.h>
#include <cmath>
#include <queue> 
using namespace std;

// Length of a given hash string.
const int HASH_LEN = 128;

// Types of message tag for a given MPI routine.
const int TASK_NONE = -1;
const int TASK_NEW = 0;
const int TASK_DONE = 1;

// Represents a given pair-wise alignment.
struct Task {
	int probNum;
	int i; // Index of the first sequence in the genes array.
	int j; // Index of the second sequence in the genes array.
};

// Result of each pair-wise alignment.
struct Result {
	int probNum;
	int probPenalty;
	char probHash[HASH_LEN + 1]; // +1 for null character.
};

MPI_Datatype createResultPacket();
MPI_Datatype createTaskPacket();
std::string getMinimumPenalties(std::string* genes, int sequencesNum, int mismatchPenalty, int gapPenalty, int* penalties);
void doSlaveTask();
int getMinimumPenalty(std::string gene1, std::string gene2, int mismatchPenalty, int gapPenalty, int* gene1Ans, int* gene2Ans);
Result getResult(std::string gene1, std::string gene2, int probNum, int mismatchPenalty, int gapPenalty);

/*
* Examples of sha512 which returns a std::string
* sw::sha512::calculate("SHA512 of std::string") // hash of a string, or
* sw::sha512::file(path) // hash of a file specified by its path, or
* sw::sha512::calculate(&data, sizeof(data)) // hash of any block of data
*/

int min3(int a, int b, int c) {
	if (a <= b && a <= c) {
		return a;
	}
	else if (b <= a && b <= c) {
		return b;
	}
	else {
		return c;
	}
}

/* Equivalent of  int *dp[width] = new int[height][width],
*  but works for width not known at compile time.
*  (Delete structure by  delete[] dp[0]; delete[] dp;).
*/
int** new2d(int width, int height) {
	int** dp = new int* [width];
	size_t size = width;
	size *= height;
	int* dp0 = new int[size];

	if (!dp || !dp0) {
		std::cerr << "getMinimumPenalty: new failed" << std::endl;
		exit(1);
	}

	dp[0] = dp0;

	for (int i = 1; i < width; i++)
		dp[i] = dp[i - 1] + height;

	return dp;
}

// Returns current wallclock time, for performance measurement.
uint64_t GetTimeStamp() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * (uint64_t)1000000 + tv.tv_usec;
}
const MPI_Comm comm = MPI_COMM_WORLD;
const int master = 0;

int main(int argc, char** argv) {
	int rank;
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	MPI_Comm_rank(comm, &rank);

	if (rank == master) {
		int mismatchPenalty;
		int gapPenalty;
		int sequencesNum;
		std::cin >> mismatchPenalty;
		std::cin >> gapPenalty;
		std::cin >> sequencesNum;
		std::string genes[sequencesNum];

		for (int i = 0; i < sequencesNum; i++)
		{
			std::cin >> genes[i];
		}
		int numPairs = sequencesNum * (sequencesNum - 1) / 2;

		int penalties[numPairs];

		uint64_t startTime = GetTimeStamp();

		// Return all the penalties and the hash of all alignments.
		std::string alignmentHash = getMinimumPenalties(genes,
			sequencesNum, mismatchPenalty, gapPenalty,
			penalties);

		// Print the time taken to do the computation.
		printf("Time: %ld us\n", (uint64_t)(GetTimeStamp() - startTime));

		// Print the alignment hash.
		std::cout << alignmentHash << std::endl;

		for (int i = 0; i < numPairs; i++) {
			std::cout << penalties[i] << " ";
		}
		std::cout << std::endl;
	}
	else {
		doSlaveTask();
	}
	MPI_Finalize();
	return 0;
}

// Initialises and returns a packet to store information about a given task.
MPI_Datatype createTaskPacket() {
	MPI_Datatype task;
	MPI_Type_contiguous(3, MPI_INT, &task);
	MPI_Type_commit(&task);
	return task;
}

// Creates and returns a packet to store the result of a given task.
MPI_Datatype createResultPacket() {
	MPI_Datatype packet;

	int blockLengths[3] = { 1, 1, HASH_LEN + 1 };

	MPI_Aint displacements[3] = {
		offsetof(Result, probNum),
		offsetof(Result, probPenalty),
		offsetof(Result, probHash)
	};

	MPI_Datatype types[3] = {
		MPI_INT,
		MPI_INT,
		MPI_CHAR
	};
	MPI_Type_create_struct(3, blockLengths, displacements, types, &packet);
	MPI_Type_commit(&packet);
	return packet;
}

/* To be called by master.
*  Distributes work and combine results to calculate the final hash which is then returned.
*/
std::string getMinimumPenalties(std::string* genes, int sequencesNum, int mismatchPenalty, int gapPenalty,
	int* penalties) {
	std::string alignmentHash = "";

	//MPI_Comm_rank(comm, &rank);
	int size;
	MPI_Comm_size(comm, &size);

	// Broadcast number of sequences, mismatch penalty, and gap penalty to all processes.
	int instanceInfo[3] = { sequencesNum, mismatchPenalty, gapPenalty };
	MPI_Bcast(instanceInfo, 3, MPI_INT, master, comm);

	// Number of pairwise alignments to be done.
	int numPairs = sequencesNum * (sequencesNum - 1) / 2;

	// Calculate length of all genetic sequences.
	int sequence_lengths[sequencesNum];
	for (int i = 0; i < sequencesNum; i++) {
		sequence_lengths[i] = genes[i].length();
	}

	// Broadcast length of all genetic sequences to all processes.
	MPI_Bcast(sequence_lengths, sequencesNum, MPI_INT, master, comm);

	// Broadcast all genetic sequences to all processes.
	for (int i = 0; i < sequencesNum; i++) {
		char sequence_buffer[sequence_lengths[i]];
		memcpy(sequence_buffer, genes[i].c_str(), sequence_lengths[i]);
		MPI_Bcast(sequence_buffer, sequence_lengths[i], MPI_CHAR, master, comm);
	}

	// Create task and result packets.
	MPI_Datatype taskPacket = createTaskPacket();
	MPI_Datatype resultPacket = createResultPacket();

	// One thread does task distribution, one thread does alignment.
	int probNum = 0;
	//omp_set_nested(1);
	#pragma omp parallel num_threads(2)
	{
		MPI_Status status;
		if (omp_get_thread_num() == 0) {
			// Queue tasks.
			queue<Task> tasks;
			for (int i = 1; i < sequencesNum; ++i) {
				for (int j = 0; j < i; ++j) {
					Task task = { probNum, i, j };
					tasks.push(task);
					probNum++;
				}
			}

			// Initial task distribution.
			for (int rank = 0; rank < size; ++rank) {

				if (!tasks.empty()) {

					Task task = tasks.front();
					MPI_Send(&task, 1, taskPacket, rank, TASK_NEW, comm);
					tasks.pop();
				}

				else  {
					Task task = { TASK_NONE, TASK_NONE, TASK_NONE };
					MPI_Send(&task, 1, taskPacket, rank, TASK_NEW, comm);
				}
			}

			Result results[numPairs];

			// Continuously distribute the remaining tasks.
			for (int i = 0; i < numPairs; i++) {
				Result result{};
				MPI_Recv(&result, 1, resultPacket, MPI_ANY_SOURCE, TASK_DONE, comm, &status);

				results[result.probNum] = result;

				if (!tasks.empty()) {
					Task task = tasks.front();
					MPI_Send(&task, 1, taskPacket, status.MPI_SOURCE, TASK_NEW, comm);
					tasks.pop();
				}
				else {
					Task task = { TASK_NONE, TASK_NONE, TASK_NONE };
					MPI_Send(&task, 1, taskPacket, status.MPI_SOURCE, TASK_NEW, comm);
				}
			}
			// Calculate final hash.
			for (int i = 0; i < numPairs; i++) {
				penalties[i] = results[i].probPenalty;
				alignmentHash = sw::sha512::calculate(alignmentHash.append(results[i].probHash));
			}
		}

		else {
			Task task;
			do {
				MPI_Recv(&task, 1, taskPacket, master, TASK_NEW, comm, &status);
				if (task.i == TASK_NONE && task.j == TASK_NONE && task.probNum == TASK_NONE) {

					break;
				}
				Result result = getResult(genes[task.i], genes[task.j], task.probNum, mismatchPenalty, gapPenalty);
				MPI_Send(&result, 1, resultPacket, master, TASK_DONE, comm);
			} while (true);
		}
	}
	return alignmentHash;
}

void doSlaveTask() {
	int size;
	MPI_Comm_size(comm, &size);

	// Receives number of genetic sequences, mismatch penalty, gap penalty of the given problem instance from master.
	int instanceInfo[3];
	MPI_Bcast(instanceInfo, 3, MPI_INT, master, comm);
	int sequencesNum = instanceInfo[0];
	int mismatchPenalty = instanceInfo[1];
	int gapPenalty = instanceInfo[2];

	// Receives genetic sequence lengths from master.
	int sequence_lengths[sequencesNum];
	MPI_Bcast(sequence_lengths, sequencesNum, MPI_INT, master, comm);

	// Receive genetic sequences from master.
	string genes[sequencesNum];
	for (int i = 0; i < sequencesNum; i++) {
		int length = sequence_lengths[i];
		char sequence_buffer[length + 1];
		MPI_Bcast(sequence_buffer, length, MPI_CHAR, master, comm);
		sequence_buffer[length] = '\0';
		genes[i] = string(sequence_buffer, length);
	}

	// Create task and result packets.
	MPI_Datatype taskPacket = createTaskPacket();
	MPI_Datatype resultPacket = createResultPacket();

	MPI_Status status;

	// Continuously work until no more tasks are given.
	Task task;
	do {
		MPI_Recv(&task, 1, taskPacket, master, TASK_NEW, comm, &status);
		if (task.i == TASK_NONE && task.j == TASK_NONE && task.probNum == TASK_NONE) {
			break;
		}

		Result result = getResult(genes[task.i], genes[task.j], task.probNum, mismatchPenalty, gapPenalty);
		MPI_Send(&result, 1, resultPacket, master, TASK_DONE, comm);
	} while (true);
}

// Calculates and returns the result of a given task.
Result getResult(std::string gene1, std::string gene2, int probNum, int mismatchPenalty, int gapPenalty) {
	int length1 = gene1.length();
	int length2 = gene2.length();
	int totalLength = length1 + length2;

	int gene1Ans[totalLength + 1], gene2Ans[totalLength + 1];

	int penalty = getMinimumPenalty(gene1, gene2, mismatchPenalty, gapPenalty, gene1Ans, gene2Ans);

	// Since we have assumed the answer to be n+m long,
	// we need to remove the extra gaps in the starting
	// id represents the index from which the arrays
	// xans, yans are useful.
	int id = 1;
	int a;
	for (a = totalLength; a >= 1; --a)
	{
		if ((char)gene1Ans[a] == '_' && (char)gene2Ans[a] == '_')
		{
			id = a + 1;
			break;
		}
	}
	std::string align1 = "";
	std::string align2 = "";
	for (a = id; a <= totalLength; a++)
	{
		align1.append(1, (char)gene1Ans[a]);
	}
	for (a = id; a <= totalLength; ++a)
	{
		align2.append(1, (char)gene2Ans[a]);
	}

	std::string align1hash = sw::sha512::calculate(align1);
	std::string align2hash = sw::sha512::calculate(align2);
	std::string problemHash = sw::sha512::calculate(align1hash.append(align2hash));

	Result result;
	result.probNum = probNum;
	result.probPenalty = penalty;
	strcpy(result.probHash, problemHash.c_str());

	return result;
}

/* Function to find out the minimum penalty.
*  Returns the minimum penalty and put the aligned sequences in xans and yans.
*/
int getMinimumPenalty(std::string gene1, std::string gene2, int mismatchPenalty, int gapPenalty, int* gene1Ans, int* gene2Ans)
{
	int i, j;

	int length1 = gene1.length();
	int length2 = gene2.length();

	int rows = length1 + 1;
	int cols = length2 + 1;

	// Table for storing optimal substructure answers.
	int** dp = new2d(length1 + 1, length2 + 1);
	size_t size = length1 + 1;
	size *= length2 + 1;
	memset(dp[0], 0, size);

	// Intialising the table.
	for (i = 0; i <= length1; ++i) {
		dp[i][0] = i * gapPenalty;
	}
	for (i = 0; i <= length2; ++i) {
		dp[0][i] = i * gapPenalty;
	}

	/* References for my parallelisation approach:
	   1. Sequential algorithm, adapted from:
			- https://www.tutorialspoint.com/zigzag-or-diagonal-traversal-of-matrix-in-cplusplus.
	   2. Parallel algorithm, idea and approach from:
			- https://etd.ohiolink.edu/apexprod/rws_etd/send_file/send?accession=kent1429528937&disposition=inline.
			- https://cse.buffalo.edu/~vipin/book_Chapters/2006/2006_2.pdf.
	*/

	int threads = omp_get_max_threads();

	// Dimensions in (matrix) cells of a given block of cells for a given thread, min = 1.
	int blockWidth = (int)ceil((1.0 * length1) / threads);
	int blockLength = (int)ceil((1.0 * length2) / threads);

	for (int traversalNum = 1; traversalNum <= (rows + cols - 1); traversalNum++)
	{
		// Column index of the starting cell of the current diagonal traversal.
		int startCol = max(1, traversalNum - threads + 1);

		// Number of cells on the current diagonal traversal.
		int cells = min(traversalNum, threads);



		omp_set_dynamic(0);
		omp_set_num_threads(threads);

		/*#pragma omp parallel for schedule(static, 1) ordered*/ // For debug

		// Each thread processes a cell on a different column along the current diagonal so no data-racing.
#pragma omp parallel for
		for (int currentCol = startCol; currentCol <= cells; currentCol++) {

			int rowStart = (currentCol - 1) * blockWidth + 1;
			int colStart = (traversalNum - currentCol) * blockLength + 1;

			// Prevents out-of-bound traversal.
			int rowEnd = min(rowStart + blockWidth, rows);
			int colEnd = min(colStart + blockLength, cols);



			// Start from cell at (currentRow, currentRol), visit its northwest, north, west neighbours.
			for (int currentRow = rowStart; currentRow < rowEnd; ++currentRow) {
				for (int currentCol = colStart; currentCol < colEnd; ++currentCol) {


					if (gene1[currentRow - 1] == gene2[currentCol - 1]) {
						dp[currentRow][currentCol] = dp[currentRow - 1][currentCol - 1];
					}
					else {
						dp[currentRow][currentCol] = min3(
							dp[currentRow - 1][currentCol - 1] + mismatchPenalty,
							dp[currentRow - 1][currentCol] + gapPenalty,
							dp[currentRow][currentCol - 1] + gapPenalty);
					}

				}
			}
		}

	}

	// Reconstructing the solution.
	int maxPossibleLength = length2 + length1;

	i = length1;
	j = length2;

	int xCoord = maxPossibleLength;
	int yCoord = maxPossibleLength;

	while (!(i == 0 || j == 0)) {
		if (gene1[i - 1] == gene2[j - 1]) {
			gene1Ans[xCoord--] = (int)gene1[i - 1];
			gene2Ans[yCoord--] = (int)gene2[j - 1];
			i--; j--;
		}

		else if (dp[i - 1][j - 1] + mismatchPenalty == dp[i][j]) {
			gene1Ans[xCoord--] = (int)gene1[i - 1];
			gene2Ans[yCoord--] = (int)gene2[j - 1];
			i--; j--;
		}

		else if (dp[i - 1][j] + gapPenalty == dp[i][j]) {
			gene1Ans[xCoord--] = (int)gene1[i - 1];
			gene2Ans[yCoord--] = (int)'_';
			i--;
		}

		else if (dp[i][j - 1] + gapPenalty == dp[i][j]) {
			gene1Ans[xCoord--] = (int)'_';
			gene2Ans[yCoord--] = (int)gene2[j - 1];
			j--;
		}
	}

	while (xCoord > 0) {
		if (i > 0) gene1Ans[xCoord--] = (int)gene1[--i];
		else gene1Ans[xCoord--] = (int)'_';
	}

	while (yCoord > 0) {
		if (j > 0) gene2Ans[yCoord--] = (int)gene2[--j];
		else gene2Ans[yCoord--] = (int)'_';
	}

	int ret = dp[length1][length2];

	delete[] dp[0];
	delete[] dp;

	return ret;
}
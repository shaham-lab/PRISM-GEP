/* Copyright (C) 2005 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.	For further
   information, see the file `LICENSE' included with this distribution. */

package cc.mallet.topics;

import java.util.Arrays;
import java.util.logging.FileHandler;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.SimpleFormatter;
import java.util.ArrayList;

import java.util.zip.*;

import java.io.*;
import java.text.NumberFormat;

import cc.mallet.types.*;
import cc.mallet.util.Randoms;

/**
 * A parallel topic model runnable task.
 * 
 * @author David Mimno, Andrew McCallum
 */

public class WorkerRunnable implements Runnable {
	
	boolean isFinished = true;

	ArrayList<TopicAssignment> data;
	int startDoc, numDocs;

	protected int numTopics; // Number of topics to be fit
	protected int doc;

	// These values are used to encode type/topic counts as
	//  count/topic pairs in a single int.
	protected int topicMask;
	protected int topicBits;

	protected int numTypes;

	protected double[] alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
	protected double alphaSum;
	protected double[] beta;   // Prior on per-topic multinomial distribution over words
	protected double betaSum;
	protected double betaScalar = 0.01; // TODO: This should be set to the number of words in the vocabulary
	public static final double DEFAULT_BETA = 0.01;
	
	protected double smoothingOnlyMass = 0.0;
	protected double[] cachedCoefficients;

	protected int[][] typeTopicCounts; // indexed by <feature index, topic index>
	protected int[] tokensPerTopic; // indexed by <topic index>

	// for dirichlet estimation
	protected int[] docLengthCounts; // histogram of document sizes
	protected int[][] topicDocCounts; // histogram of document/topic counts, indexed by <topic index, sequence position index>

	boolean shouldSaveState = false;
	boolean shouldBuildLocalCounts = true;
	
	protected Randoms random;
	
	public WorkerRunnable (int numTopics,
						   double[] alpha, double alphaSum,
						   double[] beta, double betaSum,
						   Randoms random,
						   ArrayList<TopicAssignment> data,
						   int[][] typeTopicCounts, 
						   int[] tokensPerTopic,
						   int startDoc, int numDocs) {
		

		this.data = data;

		this.numTopics = numTopics;
		this.numTypes = typeTopicCounts.length;

		if (Integer.bitCount(numTopics) == 1) {
			// exact power of 2
			topicMask = numTopics - 1;
			topicBits = Integer.bitCount(topicMask);
		}
		else {
			// otherwise add an extra bit
			topicMask = Integer.highestOneBit(numTopics) * 2 - 1;
			topicBits = Integer.bitCount(topicMask);
		}

		this.typeTopicCounts = typeTopicCounts;
		this.tokensPerTopic = tokensPerTopic;
		
		this.alphaSum = alphaSum;
		this.alpha = alpha;
		this.beta = beta;
		this.betaSum = betaSum;
		this.random = random;
		
		this.startDoc = startDoc;
		this.numDocs = numDocs;

		cachedCoefficients = new double[ numTopics ];
		// saveTokensPerTopicToFile(); // Save the tokens per topic to a file for debugging
		// System.exit(0); // will remove this line later

		//System.err.println("WorkerRunnable Thread: " + numTopics + " topics, " + topicBits + " topic bits, " + 
		//				   Integer.toBinaryString(topicMask) + " topic mask");

	}

	/**
	 *  If there is only one thread, we don't need to go through 
	 *   communication overhead. This method asks this worker not
	 *   to prepare local type-topic counts. The method should be
	 *   called when we are using this code in a non-threaded environment.
	 */
	public void makeOnlyThread() {
		shouldBuildLocalCounts = false;
	}

	public int[] getTokensPerTopic() { return tokensPerTopic; }
	public int[][] getTypeTopicCounts() { return typeTopicCounts; }

	public int[] getDocLengthCounts() { return docLengthCounts; }
	public int[][] getTopicDocCounts() { return topicDocCounts; }

	public void initializeAlphaStatistics(int size) {
		docLengthCounts = new int[size];
		topicDocCounts = new int[numTopics][size];
	}
	
	public void collectAlphaStatistics() {
		shouldSaveState = true;
	}

	public void resetBeta(double[] beta, double betaSum) {
		this.beta = beta;
		this.betaSum = betaSum;
	}

	/**
	 *  Once we have sampled the local counts, trash the 
	 *   "global" type topic counts and reuse the space to 
	 *   build a summary of the type topic counts specific to 
	 *   this worker's section of the corpus.
	 */
	public void buildLocalTypeTopicCounts () {

		// Clear the topic totals
		Arrays.fill(tokensPerTopic, 0);

		// Clear the type/topic counts, only 
		//  looking at the entries before the first 0 entry.

		for (int type = 0; type < typeTopicCounts.length; type++) {

			int[] topicCounts = typeTopicCounts[type];
			
			int position = 0;
			while (position < topicCounts.length && 
				   topicCounts[position] > 0) {
				topicCounts[position] = 0;
				position++;
			}
		}

        for (doc = startDoc;
			 doc < data.size() && doc < startDoc + numDocs;
             doc++) {

			TopicAssignment document = data.get(doc);

            FeatureSequence tokens = (FeatureSequence) document.instance.getData();
            FeatureSequence topicSequence =  (FeatureSequence) document.topicSequence;

            int[] topics = topicSequence.getFeatures();
            for (int position = 0; position < tokens.size(); position++) {

				int topic = topics[position];

				if (topic == ParallelTopicModel.UNASSIGNED_TOPIC) { continue; }

				tokensPerTopic[topic]++;
				
				// The format for these arrays is 
				//  the topic in the rightmost bits
				//  the count in the remaining (left) bits.
				// Since the count is in the high bits, sorting (desc)
				//  by the numeric value of the int guarantees that
				//  higher counts will be before the lower counts.
				
				int type = tokens.getIndexAtPosition(position);

				int[] currentTypeTopicCounts = typeTopicCounts[ type ];
				
				// Start by assuming that the array is either empty
				//  or is in sorted (descending) order.
				
				// Here we are only adding counts, so if we find 
				//  an existing location with the topic, we only need
				//  to ensure that it is not larger than its left neighbor.
				
				int index = 0;
				int currentTopic = currentTypeTopicCounts[index] & topicMask;
				int currentValue;
				
				while (currentTypeTopicCounts[index] > 0 && currentTopic != topic) {
					index++;
					if (index == currentTypeTopicCounts.length) {
						System.out.println("overflow on type " + type);
					}
					currentTopic = currentTypeTopicCounts[index] & topicMask;
				}
				currentValue = currentTypeTopicCounts[index] >> topicBits;
				
				if (currentValue == 0) {
					// new value is 1, so we don't have to worry about sorting
					//  (except by topic suffix, which doesn't matter)
					
					currentTypeTopicCounts[index] =
						(1 << topicBits) + topic;
				}
				else {
					currentTypeTopicCounts[index] =
						((currentValue + 1) << topicBits) + topic;
					
					// Now ensure that the array is still sorted by 
					//  bubbling this value up.
					while (index > 0 &&
						   currentTypeTopicCounts[index] > currentTypeTopicCounts[index - 1]) {
						int temp = currentTypeTopicCounts[index];
						currentTypeTopicCounts[index] = currentTypeTopicCounts[index - 1];
						currentTypeTopicCounts[index - 1] = temp;
						
						index--;
					}
				}
			}
		}

	}

	public void saveTopicsToFile(int docId, int[] topicAssignments) {
		try (FileWriter fileWriter = new FileWriter("AfterGibbs_doc_topics_debug.txt", true);
			 PrintWriter printWriter = new PrintWriter(fileWriter)) {
	
			printWriter.println("Document " + docId + " Topic Assignments:");
	
			for (int i = 0; i < topicAssignments.length; i++) {
				printWriter.printf("Token %d -> Topic %d%n", i, topicAssignments[i]);
			}
	
			printWriter.println("#########################\n");
	
		} catch (IOException e) {
			System.err.println("Error writing topic assignments to file: " + e.getMessage());
		}
	}

	// save tokenPerTopic to file
	public void saveTokensPerTopicToFile() {
		try (FileWriter fileWriter = new FileWriter("tokens_per_topic.txt", true);
			 PrintWriter printWriter = new PrintWriter(fileWriter)) {
	
			printWriter.println("Tokens Per Topic:");
	
			for (int i = 0; i < tokensPerTopic.length; i++) {
				printWriter.printf("Topic %d -> Count %d%n", i, tokensPerTopic[i]);
			}
	
			printWriter.println("#########################\n");
	
		} catch (IOException e) {
			System.err.println("Error writing tokens per topic to file: " + e.getMessage());
		}
	}

	/**
	 * Collapsed Gibbs Formula for topic conditional probability expression:
	 * 		P(z_i = k | ·) ∝ (n_{d,k} + α_k) * (n_{k,w} + β_w) / (n_k + β * W)
	 * 
	 * This probability has two parts:
	 * - Document-topic factor: (n_{d,k} + α_k)
	 * - Topic-word factor: (n_{k,w} + β_w) / (n_k + β * W)
	 * 
	 * run ()
	 * 
	 * Params:
	 * 
	 * 1. smoothingOnlyMass - precomputes α_k * β_w / ( n_k + β * W ) mass.
	 * 		This is the pure smoothing (prior) probability contribution to assigning topic k to word w, in the absence of any actual word or topic usage counts.
	 * 		Used for efficiency, so it doesn’t need to be recomputed repeatedly during sampling for each token.
	 * 
	 * 2. cachedCoefficients - cached coefficient α_k / ( n_k + β * W ). // TODO
	 * 		An array stores one value per topic, representing the α (document-topic prior) scaled by the normalizing denominator of the topic-word distribution.
	 * 		During Gibbs sampling, this is reused frequently when computing probabilities for topics not yet active in a document.
	 * 
	 * 
	 * 3. tokenSequence - A sequence of integers, each representing a word (by type ID) in the document.
	 * 
	 * 4. topicSequence - A sequence of integers, each representing a topic (by topic ID) assigned to the corresponding word in the document.
	 * 
	 * 
	 **/
	public void run () {

		try {
			
			if (! isFinished) { System.out.println("already running!"); return; }
			
			isFinished = false;
			
			// Initialize the smoothing-only sampling bucket
			smoothingOnlyMass = 0;
			
			// Initialize the cached coefficients, using only smoothing.
			//  These values will be selectively replaced in documents with
			//  non-zero counts in particular topics.

			for (int topic=0; topic < numTopics; topic++) {
				smoothingOnlyMass += alpha[topic] * betaScalar / (tokensPerTopic[topic] + betaSum);
				// cachedCoefficients[topic] =  alpha[topic] / (tokensPerTopic[topic] + betaSum);
			}
			// System.out.println("smoothingOnlyMass: " + smoothingOnlyMass);

			for (doc = startDoc;
				 doc < data.size() && doc < startDoc + numDocs;
				 doc++) {
				
				/*
				  if (doc % 10000 == 0) {
				  System.out.println("processing doc " + doc);
				  }
				*/
				
				FeatureSequence tokenSequence =
					(FeatureSequence) data.get(doc).instance.getData();
				LabelSequence topicSequence =
					(LabelSequence) data.get(doc).topicSequence;
				

				sampleTopicsForOneDoc (tokenSequence, topicSequence,
									   true);
			}
			// exit program for checking
			// System.exit(0);
			
			if (shouldBuildLocalCounts) {
				buildLocalTypeTopicCounts();
			}

			shouldSaveState = true;
			isFinished = true;

		} catch (Exception e) {
			isFinished = true;
			e.printStackTrace();
		}

	}

	/**
	 * Iterates over each word in a document.
	 * Resamples the topic for each word using the probability:
	 * 
	 * 		P( z_i=k | w_i=v,d ) ∝ (n_d,k + alpha_k) * (n_k,v + beta) / (n_k + betaSum)
	 * 
	 * where n_d,k is the number of times words in document d were samples from topic k,
	 * n_d is the number of words in document d,
	 * n_k,v is the number of times word type v was sampled from topic k,
	 * 
	 * The resampled topic updates typeTopicCounts and tokensPerTopic.
	 */
	protected void sampleTopicsForOneDoc (FeatureSequence tokenSequence,
										  FeatureSequence topicSequence,
										  boolean readjustTopicsAndStats /* currently ignored */) {


		// Extracts the current topic assignments for each word in the document.
		int[] oneDocTopics = topicSequence.getFeatures();
		// saveTopicsToFile(doc, oneDocTopics); // Save the topic assignments to a file for debugging

		// for (int i = 0; i < oneDocTopics.length; i++) {
        //     logger.fine("oneDocTopics " + i + ": " + oneDocTopics[i]);
        // }
        
        // logger.fine("\n#########################\n");

		int[] currentTypeTopicCounts;
		int type, oldTopic, newTopic;
		double topicWeightsSum; // never used
		int docLength = tokenSequence.getLength();

		// Initialize Local Topic Counts
		int[] localTopicCounts = new int[numTopics]; // localTopicCounts - n_{d,k} Counts occurrences of each topic in the document
		int[] localTopicIndex = new int[numTopics]; // localTopicIndex - Array to store the indices of non-zero topics

		//		populate topic counts
		for (int position = 0; position < docLength; position++) {
			if (oneDocTopics[position] == ParallelTopicModel.UNASSIGNED_TOPIC) { continue; }
			localTopicCounts[oneDocTopics[position]]++; // localTopicCounts - Counts occurrences of each topic in the document
		}

		// Build an array that densely lists the topics that
		//  have non-zero counts.
		int denseIndex = 0;
		for (int topic = 0; topic < numTopics; topic++) {
			if (localTopicCounts[topic] != 0) {
				localTopicIndex[denseIndex] = topic;
				denseIndex++;
			}
		}

		// Record the total number of non-zero topics
		int nonZeroTopics = denseIndex;

		//		Initialize the topic count/beta sampling bucket

		// Accumulates the unnormalized probability mass from document-topic counts (n_{d,k}),
		// assuming a uniform likelihood over words via the β prior.
		// It does not depend on the current word but still includes β to ensure the correct
		// scaling of probabilities for unseen word-topic pairs.
		double topicBetaMass = 0.0; // TODO: Should be  a vector size topics

		// Initialize cached coefficients and the topic/beta 
		//  normalizing constant.

		for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
			int topic = localTopicIndex[denseIndex];
			int n = localTopicCounts[topic]; // n_{d,k} - number of words in document d assigned to topic k

			//	initialize the normalization constant for the (B * n_{t|d}) term
			// This is the complementary formula of the smoothingOnlyMass.
			// topicBetaMass += betaScalar * n /	(tokensPerTopic[topic] + betaSum);	

			//	update the coefficients for the non-zero topics
			cachedCoefficients[topic] =	(alpha[topic] + n) / (tokensPerTopic[topic] + betaSum);
		}

		double topicTermMass = 0.0;

		double[] topicTermScores = new double[numTopics]; // Array stores unnormalized probability scores for topics with non-zero word-topic counts for the current word type.
		int[] topicTermIndices; // never used
		int[] topicTermValues; // never used
		int i;
		double score; // Unnormalized probability score for the current topic.

		//	Iterate over the positions (words) in the document  - Remove Word from Counts (Before Resampling)
		for (int position = 0; position < docLength; position++) {

			type = tokenSequence.getIndexAtPosition(position);
			oldTopic = oneDocTopics[position];

			betaScalar = beta[type]; // beta_w

			smoothingOnlyMass = 0.0;
			for (int topic = 0; topic < numTopics; topic++) {
				smoothingOnlyMass += alpha[topic] * betaScalar / (tokensPerTopic[topic] + betaSum);
			}

			topicBetaMass = 0.0; // TODO: Should be  a vector size topics
			for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
				int topic = localTopicIndex[denseIndex];
				int n = localTopicCounts[topic]; // n_{d,k} - number of words in document d assigned to topic k
	
				//	initialize the normalization constant for the (B * n_{t|d}) term
				// This is the complementary formula of the smoothingOnlyMass.
				topicBetaMass += betaScalar * n /	(tokensPerTopic[topic] + betaSum);	
			}

			currentTypeTopicCounts = typeTopicCounts[type]; // n_{k,w}
			
			if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {
				//	Remove this token from all counts. 
				
				// Remove this topic's contribution to the 
				//  normalizing constants
				smoothingOnlyMass -= alpha[oldTopic] * betaScalar / 
					(tokensPerTopic[oldTopic] + betaSum);
				topicBetaMass -= betaScalar * localTopicCounts[oldTopic] /
					(tokensPerTopic[oldTopic] + betaSum);
				
				// Decrement the local doc/topic counts
				localTopicCounts[oldTopic]--; // n_{d,k} - 1
				
				// Maintain the dense index, if we are deleting
				//  the old topic
				if (localTopicCounts[oldTopic] == 0) {
					
					// First get to the dense location associated with
					//  the old topic.
					
					denseIndex = 0;
					
					// We know it's in there somewhere, so we don't 
					//  need bounds checking.
					while (localTopicIndex[denseIndex] != oldTopic) {
						denseIndex++;
					}
				
					// shift all remaining dense indices to the left.
					while (denseIndex < nonZeroTopics) {
						if (denseIndex < localTopicIndex.length - 1) {
							localTopicIndex[denseIndex] = 
								localTopicIndex[denseIndex + 1];
						}
						denseIndex++;
					}
					
					nonZeroTopics --;
				}

				// Decrement the global topic count totals
				tokensPerTopic[oldTopic]--; // n_k - 1
				assert(tokensPerTopic[oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";
			

				// Add the old topic's contribution back into the
				//  normalizing constants.
				smoothingOnlyMass += alpha[oldTopic] * betaScalar / 
					(tokensPerTopic[oldTopic] + betaSum);
				topicBetaMass += betaScalar * localTopicCounts[oldTopic] /
					(tokensPerTopic[oldTopic] + betaSum);

				// Reset the cached coefficient for this topic
				cachedCoefficients[oldTopic] = 
					(alpha[oldTopic] + localTopicCounts[oldTopic]) /
					(tokensPerTopic[oldTopic] + betaSum);
			}


			// Now go over the type/topic counts, decrementing
			//  where appropriate, and calculating the score
			//  for each topic at the same time.

			int index = 0;
			int currentTopic, currentValue; 
			// currentTopic - Topic ID for the current word-topic entry. 
			// currentValue - n_{w,k} how many times the current word type has been assigned to this topic.

			boolean alreadyDecremented = (oldTopic == ParallelTopicModel.UNASSIGNED_TOPIC);

			// stores the unnormalized probability mass contributed by word-topic co-occurrence counts 
			// (i.e., how often the current word type appears in all topics in total).
			// This is the sum of the scores for all topics with non-zero word-topic counts.
			// It is used to sample a new topic for the current word type.
			topicTermMass = 0.0; 

			while (index < currentTypeTopicCounts.length && 
				   currentTypeTopicCounts[index] > 0) {
				currentTopic = currentTypeTopicCounts[index] & topicMask;
				currentValue = currentTypeTopicCounts[index] >> topicBits;

				if (! alreadyDecremented && 
					currentTopic == oldTopic) {

					// We're decrementing and adding up the 
					//  sampling weights at the same time, but
					//  decrementing may require us to reorder
					//  the topics, so after we're done here,
					//  look at this cell in the array again.

					currentValue --; // n_{w,k} - 1
					if (currentValue == 0) {
						currentTypeTopicCounts[index] = 0;
					}
					else {
						currentTypeTopicCounts[index] =
							(currentValue << topicBits) + oldTopic;
					}
					
					// Shift the reduced value to the right, if necessary.

					int subIndex = index;
					while (subIndex < currentTypeTopicCounts.length - 1 && 
						   currentTypeTopicCounts[subIndex] < currentTypeTopicCounts[subIndex + 1]) {
						int temp = currentTypeTopicCounts[subIndex];
						currentTypeTopicCounts[subIndex] = currentTypeTopicCounts[subIndex + 1];
						currentTypeTopicCounts[subIndex + 1] = temp;
						
						subIndex++;
					}

					alreadyDecremented = true;
				}
				else {
					score = 
						cachedCoefficients[currentTopic] * currentValue;
					topicTermMass += score;
					topicTermScores[index] = score;

					index++;
				}
			}


			// Sample a New Topic
			// draws a random number from a multinomial distribution over topics, where:
			// Each topic's weight (probability) is unnormalized and split across three components:
			// 1. (smoothingOnlyMass) -> prior-only fallback - Topics not seen in the doc or with this word (n_{d,k} = 0, n_{k,w} = 0)
			// 2. (topicBetaMass) -> Topics used in the doc but not with this word (n_{d,k} > 0, n_{k,w} = 0)
			// 3. (topicTermMass) -> Topics where the word has been seen before (n_{k,w} > 0)

			double sample = random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass); // topicTermMass + topicBetaMass + smoothingOnlyMass -> total mass of the multinomial distribution we’re sampling from.

			// sample - a number between 0 and totalMass, selecting a point in the combined probability space.
			// number is later used to decide which topic to assign to the current word (based on which region it lands in).
			double origSample = sample;

			//	Make sure it actually gets set
			newTopic = -1;

			// Now we check which region this sample landed in

			if (sample < topicTermMass) {
				// falling into the topicTermMass region

				//topicTermCount++;

			
				// We need to find the first topic whose score is less than the sample value.
				//
				// 0           p0        p0+p1     p0+p1+p2         ...      total
				// |-----------|----------|-----------|-----------------------|
				// Topic 0     Topic 1    Topic 2     Topic 3                Topic K-1
				// 
				// This number line is a partitioning of the total mass into contiguous segments — each one proportional to p_k.
				// So the position of the sample tells us which topic's segment it falls into.
				// We can find the topic by subtracting the scores from the sample until we find a negative value.
				// The index of the last topic whose score was subtracted is the topic we want.

				i = -1;
				while (sample > 0) {
					i++;
					sample -= topicTermScores[i];
				}

				newTopic = currentTypeTopicCounts[i] & topicMask;
				currentValue = currentTypeTopicCounts[i] >> topicBits;
				
				currentTypeTopicCounts[i] = ((currentValue + 1) << topicBits) + newTopic; // n_{k,w} + 1

				// Bubble the new value up, if necessary
				
				while (i > 0 &&
					   currentTypeTopicCounts[i] > currentTypeTopicCounts[i - 1]) {
					int temp = currentTypeTopicCounts[i];
					currentTypeTopicCounts[i] = currentTypeTopicCounts[i - 1];
					currentTypeTopicCounts[i - 1] = temp;

					i--;
				}

			}
			else {
				sample -= topicTermMass;

				if (sample < topicBetaMass) {
					// falling into the topicBetaMass region
					//betaTopicCount++;

					sample /= betaScalar;

					// Reminder: 
					// localTopicIndex - indexes of topics used in this doc
					// localTopicCounts - counts per topic used in this doc

					for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
						int topic = localTopicIndex[denseIndex];

						// This subtracts each topic's contribution to the topicBetaMass
						// until the sample becomes ≤ 0.
						// This means that the sample is now in the range of the topicBetaMass region.
						// The topic that caused the sample to go negative is the new topic.
						sample -= localTopicCounts[topic] /
							(tokensPerTopic[topic] + betaSum);

						if (sample <= 0.0) {
							newTopic = topic;
							break;
						}
					}

				}
				else {
					//smoothingOnlyCount++;

					sample -= topicBetaMass;

					sample /= betaScalar;

					newTopic = 0;
					sample -= alpha[newTopic] /
						(tokensPerTopic[newTopic] + betaSum);

					while (sample > 0.0) {
						newTopic++;
						sample -= alpha[newTopic] / 
							(tokensPerTopic[newTopic] + betaSum);
					}
					
				}

				// Move to the position for the new topic,
				//  which may be the first empty position if this
				//  is a new topic for this word.
				
				index = 0;
				while (currentTypeTopicCounts[index] > 0 &&
					   (currentTypeTopicCounts[index] & topicMask) != newTopic) {
					index++;
					if (index == currentTypeTopicCounts.length) {
						System.err.println("type: " + type + " new topic: " + newTopic);
						for (int k=0; k<currentTypeTopicCounts.length; k++) {
							System.err.print((currentTypeTopicCounts[k] & topicMask) + ":" + 
											 (currentTypeTopicCounts[k] >> topicBits) + " ");
						}
						System.err.println();

					}
				}


				// index should now be set to the position of the new topic,
				//  which may be an empty cell at the end of the list.

				if (currentTypeTopicCounts[index] == 0) {
					// inserting a new topic, guaranteed to be in
					//  order w.r.t. count, if not topic.
					currentTypeTopicCounts[index] = (1 << topicBits) + newTopic;
				}
				else {
					currentValue = currentTypeTopicCounts[index] >> topicBits;
					currentTypeTopicCounts[index] = ((currentValue + 1) << topicBits) + newTopic;

					// Bubble the increased value left, if necessary
					while (index > 0 &&
						   currentTypeTopicCounts[index] > currentTypeTopicCounts[index - 1]) {
						int temp = currentTypeTopicCounts[index];
						currentTypeTopicCounts[index] = currentTypeTopicCounts[index - 1];
						currentTypeTopicCounts[index - 1] = temp;

						index--;
					}
				}

			}

			if (newTopic == -1) {
				System.err.println("WorkerRunnable sampling error: "+ origSample + " " + sample + " " + smoothingOnlyMass + " " + 
						topicBetaMass + " " + topicTermMass);
				newTopic = numTopics-1; // TODO is this appropriate
				//throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
			}
			//assert(newTopic != -1);


			// Update Counts with the ## New Topic ##
			
			//			Put that new topic into the counts
			oneDocTopics[position] = newTopic; // z_i = k

			smoothingOnlyMass -= alpha[newTopic] * betaScalar / 
				(tokensPerTopic[newTopic] + betaSum);
			topicBetaMass -= betaScalar * localTopicCounts[newTopic] /
				(tokensPerTopic[newTopic] + betaSum);

			localTopicCounts[newTopic]++; // n_{d,k} + 1

			// If this is a new topic for this document,
			//  add the topic to the dense index.
			if (localTopicCounts[newTopic] == 1) {
				
				// First find the point where we 
				//  should insert the new topic by going to
				//  the end (which is the only reason we're keeping
				//  track of the number of non-zero
				//  topics) and working backwards

				denseIndex = nonZeroTopics;

				while (denseIndex > 0 &&
					   localTopicIndex[denseIndex - 1] > newTopic) {

					localTopicIndex[denseIndex] =
						localTopicIndex[denseIndex - 1];
					denseIndex--;
				}
				
				localTopicIndex[denseIndex] = newTopic;
				nonZeroTopics++;
			}

			tokensPerTopic[newTopic]++; // n_k + 1

			//	update the coefficients for the non-zero topics
			cachedCoefficients[newTopic] =
				(alpha[newTopic] + localTopicCounts[newTopic]) /
				(tokensPerTopic[newTopic] + betaSum);

			smoothingOnlyMass += alpha[newTopic] * betaScalar / 
				(tokensPerTopic[newTopic] + betaSum);
			topicBetaMass += betaScalar * localTopicCounts[newTopic] /
				(tokensPerTopic[newTopic] + betaSum);

		}

		// saveTopicsToFile(doc, oneDocTopics); // Save the topic assignments to a file for debugging

		if (shouldSaveState) {
			// Update the document-topic count histogram,
			//  for dirichlet estimation
			docLengthCounts[ docLength ]++;

			for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
				int topic = localTopicIndex[denseIndex];
				
				topicDocCounts[topic][ localTopicCounts[topic] ]++;
			}
		}

		//	Clean up our mess: reset the coefficients to values with only
		//	smoothing. The next doc will update its own non-zero topics...

		for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
			int topic = localTopicIndex[denseIndex];

			cachedCoefficients[topic] =
				alpha[topic] / (tokensPerTopic[topic] + betaSum);
		}

	}

}

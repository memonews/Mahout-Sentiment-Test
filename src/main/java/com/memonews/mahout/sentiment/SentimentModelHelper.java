/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.memonews.mahout.sentiment;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.text.SimpleDateFormat;
import java.util.Collection;
import java.util.Locale;
import java.util.Random;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import com.google.common.base.Charsets;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.io.Closeables;
import com.google.common.io.Files;

final class SentimentModelHelper {

    private static final SimpleDateFormat[] DATE_FORMATS = { new SimpleDateFormat("", Locale.ENGLISH), new SimpleDateFormat("MMM-yyyy", Locale.ENGLISH),
	    new SimpleDateFormat("dd-MMM-yyyy HH:mm:ss", Locale.ENGLISH) };
    public static final int FEATURES = 10000;
    // 1997-01-15 00:01:00 GMT
    private static final long DATE_REFERENCE = 853286460;
    private static final long MONTH = 30 * 24 * 3600;
    private static final long WEEK = 7 * 24 * 3600;

    private final Random rand = RandomUtils.getRandom();
    private final Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_31);
    private final FeatureVectorEncoder encoder = new StaticWordValueEncoder("body");
    private final FeatureVectorEncoder bias = new ConstantValueEncoder("Intercept");

    FeatureVectorEncoder getEncoder() {
	return encoder;
    }

    FeatureVectorEncoder getBias() {
	return bias;
    }

    Random getRandom() {
	return rand;
    }

    Vector encodeFeatureVector(final File file, final Multiset<String> overallCounts) throws IOException {
	final Multiset<String> words = ConcurrentHashMultiset.create();

	final BufferedReader reader = Files.newReader(file, Charsets.UTF_8);
	try {
	    countWords(analyzer, words, reader, overallCounts);
	} finally {
	    Closeables.closeQuietly(reader);
	}

	final Vector v = new RandomAccessSparseVector(FEATURES);
	bias.addToVector("", 1, v);
	for (final String word : words.elementSet()) {
	    encoder.addToVector(word, Math.log1p(words.count(word)), v);
	}

	return v;
    }

    private static void countWords(final Analyzer analyzer, final Collection<String> words, final Reader in, final Multiset<String> overallCounts) throws IOException {
	final TokenStream ts = analyzer.reusableTokenStream("text", in);
	ts.addAttribute(CharTermAttribute.class);
	ts.reset();
	while (ts.incrementToken()) {
	    final String s = ts.getAttribute(CharTermAttribute.class).toString();
	    words.add(s);
	}
	overallCounts.addAll(words);
    }
}

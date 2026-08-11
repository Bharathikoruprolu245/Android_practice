package com.biomedixai.app

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.net.HttpURLConnection
import java.net.URL
import org.json.JSONObject
import java.util.Locale

class MainActivity : AppCompatActivity() {

    /*
     * Because we are using:
     *
     * adb reverse tcp:8000 tcp:8000
     *
     * the Android phone can access the
     * FastAPI server using localhost.
     */
    private val BASE_URL = "http://127.0.0.1:8000"

    private lateinit var diseaseInput: EditText
    private lateinit var speciesSpinner: Spinner
    private lateinit var guideRnaInput: EditText

    private lateinit var discoverButton: Button
    private lateinit var druggabilityButton: Button
    private lateinit var crisprButton: Button
    private lateinit var analyzeButton: Button

    private lateinit var statusText: TextView
    private lateinit var resultText: TextView

    // Stores the hub gene returned by Target Discovery
    private var hubGene: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        diseaseInput = findViewById(R.id.diseaseInput)
        speciesSpinner = findViewById(R.id.speciesSpinner)
        guideRnaInput = findViewById(R.id.guideRnaInput)

        discoverButton = findViewById(R.id.discoverButton)
        druggabilityButton = findViewById(R.id.druggabilityButton)
        crisprButton = findViewById(R.id.crisprButton)
        analyzeButton = findViewById(R.id.analyzeButton)

        statusText = findViewById(R.id.statusText)
        resultText = findViewById(R.id.resultText)

        setupSpeciesDropdown()

        // -----------------------------
        // TARGET DISCOVERY
        // -----------------------------

        discoverButton.setOnClickListener {

            val disease = diseaseInput.text.toString().trim()
            val species = speciesSpinner.selectedItem.toString()

            if (disease.isEmpty()) {

                showError("Please enter a disease name")

                return@setOnClickListener
            }

            val json = """
                {
                    "disease_name": "$disease",
                    "species": "$species"
                }
            """.trimIndent()

            callApi(
                endpoint = "/api/targets",
                json = json,
                type = "targets"
            )
        }

        // -----------------------------
        // DRUGGABILITY
        // -----------------------------

        druggabilityButton.setOnClickListener {

            val gene = hubGene

            if (gene == null) {

                showError(
                    "First click Discover Targets to obtain the hub gene."
                )

                return@setOnClickListener
            }

            val json = """
                {
                    "gene_symbol": "$gene"
                }
            """.trimIndent()

            callApi(
                endpoint = "/api/druggability",
                json = json,
                type = "druggability"
            )
        }

        // -----------------------------
        // CRISPR
        // -----------------------------

        crisprButton.setOnClickListener {

            val gene = hubGene

            if (gene == null) {

                showError(
                    "First click Discover Targets to obtain the hub gene."
                )

                return@setOnClickListener
            }

            val guideRna =
                guideRnaInput.text.toString().trim()

            if (guideRna.isEmpty()) {

                showError(
                    "Enter a guide RNA for CRISPR analysis."
                )

                return@setOnClickListener
            }

            val species =
                speciesSpinner.selectedItem.toString()

            val json = """
                {
                    "gene_symbol": "$gene",
                    "guide_rna": "$guideRna",
                    "species": "$species",
                    "max_mismatches": 6
                }
            """.trimIndent()

            callApi(
                endpoint = "/api/crispr",
                json = json,
                type = "crispr"
            )
        }

        // -----------------------------
        // COMPLETE ANALYSIS
        // -----------------------------

        analyzeButton.setOnClickListener {

            val disease =
                diseaseInput.text.toString().trim()

            val species =
                speciesSpinner.selectedItem.toString()

            val guideRna =
                guideRnaInput.text.toString().trim()

            if (disease.isEmpty()) {

                showError(
                    "Please enter a disease name."
                )

                return@setOnClickListener
            }

            if (guideRna.isEmpty()) {

                showError(
                    "Enter a guide RNA for Complete Analysis."
                )

                return@setOnClickListener
            }

            val json = """
                {
                    "disease_name": "$disease",
                    "species": "$species",
                    "guide_rna": "$guideRna",
                    "gene_limit": 10,
                    "centrality_method": "degree",
                    "max_mismatches": 6
                }
            """.trimIndent()

            callApi(
                endpoint = "/api/analyze",
                json = json,
                type = "analyze"
            )
        }
    }

    // ============================================================
    // SPECIES DROPDOWN
    // ============================================================

    private fun setupSpeciesDropdown() {

        val species = arrayOf(
            "human",
            "mouse",
            "rice",
            "arabidopsis",
            "yeast",
            "zebrafish",
            "fruit fly",
            "e. coli"
        )

        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            species
        )

        adapter.setDropDownViewResource(
            android.R.layout.simple_spinner_dropdown_item
        )

        speciesSpinner.adapter = adapter
    }

    // ============================================================
    // API CALL
    // ============================================================

    private fun callApi(
        endpoint: String,
        json: String,
        type: String
    ) {

        statusText.text = "Running analysis..."
        resultText.text = ""

        CoroutineScope(Dispatchers.IO).launch {

            try {

                val url =
                    URL(BASE_URL + endpoint)

                val connection =
                    url.openConnection()
                            as HttpURLConnection

                connection.requestMethod = "POST"

                connection.setRequestProperty(
                    "Content-Type",
                    "application/json"
                )

                connection.setRequestProperty(
                    "Accept",
                    "application/json"
                )

                connection.doOutput = true

                connection.outputStream.use { output ->

                    output.write(
                        json.toByteArray(Charsets.UTF_8)
                    )
                }

                val responseCode =
                    connection.responseCode

                val response =

                    if (responseCode in 200..299) {

                        connection.inputStream
                            .bufferedReader()
                            .use { it.readText() }

                    } else {

                        connection.errorStream
                            ?.bufferedReader()
                            ?.use { it.readText() }
                            ?: "Unknown server error"
                    }

                connection.disconnect()

                withContext(Dispatchers.Main) {

                    if (responseCode in 200..299) {

                        statusText.text =
                            "API Success ✅"

                        /*
                         * Target Discovery response contains:
                         *
                         * hub_gene_symbol
                         *
                         * Save it so Druggability and
                         * CRISPR can use it automatically.
                         */
                        if (type == "targets") {

                            hubGene =
                                extractHubGene(response)

                            if (hubGene != null) {

                                statusText.text =
                                    "Targets Found ✅  Hub Gene: $hubGene"
                            }
                        }

                        resultText.text =
                            formatResult(response)
                    }

                    else {

                        statusText.text =
                            "API Error ❌ ($responseCode)"

                        resultText.text =
                            response
                    }
                }

            } catch (e: Exception) {

                withContext(Dispatchers.Main) {

                    statusText.text =
                        "Connection Failed ❌"

                    resultText.text =
                        e.message ?: "Unknown connection error"
                }
            }
        }
    }

    // ============================================================
    // EXTRACT HUB GENE
    // ============================================================

    private fun extractHubGene(
        response: String
    ): String? {

        val key =
            "\"hub_gene_symbol\""

        val index =
            response.indexOf(key)

        if (index == -1) {
            return null
        }

        val colon =
            response.indexOf(
                ":",
                index
            )

        if (colon == -1) {
            return null
        }

        val start =
            response.indexOf(
                "\"",
                colon + 1
            )

        if (start == -1) {
            return null
        }

        val end =
            response.indexOf(
                "\"",
                start + 1
            )

        if (end == -1) {
            return null
        }

        return response.substring(
            start + 1,
            end
        )
    }

    // ============================================================
    // SIMPLE RESULT FORMAT
    // ============================================================

    private fun formatResult(response: String): String {
        return try {
            val root = JSONObject(response)

            val result = root.optJSONObject("result")

            if (result == null) {
                return response
            }

            val disease = result.optString("disease_name", "Unknown")
            val hubGene = result.optString("hub_gene_symbol", "Unknown")

            val centrality = result.optJSONObject("centrality_scores")

            val output = StringBuilder()

            output.append("🎯 TARGET DISCOVERY\n\n")

            output.append("Disease\n")
            output.append("$disease\n\n")

            output.append("Hub Gene\n")
            output.append("🧬 $hubGene\n\n")

            output.append("Target Genes\n")

            if (centrality != null) {

                val genes = mutableListOf<Pair<String, Double>>()

                val keys = centrality.keys()

                while (keys.hasNext()) {
                    val gene = keys.next()
                    val score = centrality.optDouble(gene, 0.0)

                    genes.add(Pair(gene, score))
                }

                // Highest score first
                genes.sortByDescending { it.second }

                for ((gene, score) in genes) {

                    output.append(
                        String.format(
                            Locale.US,
                            "%-8s %.2f\n",
                            gene,
                            score
                        )
                    )
                }
            }

            output.append("\n✅ Target discovery completed")

            output.toString()

        } catch (e: Exception) {

            "Unable to format result:\n${e.message}\n\n$response"
        }
    }

    // ============================================================
    // ERROR
    // ============================================================

    private fun showError(
        message: String
    ) {

        statusText.text =
            "Error ❌"

        resultText.text =
            message
    }
}